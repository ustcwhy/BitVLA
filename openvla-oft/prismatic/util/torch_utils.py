"""
torch_utils.py

General utilities for randomness, mixed precision training, and miscellaneous checks in PyTorch.

Random `set_global_seed` functionality is taken directly from PyTorch-Lighting:
    > Ref: https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/utilities/seed.py

This is pretty important to get right if we're every randomly generating our masks (or prefix dropout) inside our
Dataset __getitem__() with multiple workers... if not handled properly, we will get repeated augmentations anytime
we inject randomness from non-PyTorch sources (e.g., numpy, random)!
    > Ref: https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/

Terminology
    -> World Size :: Total number of processes distributed over (# nodes x # devices) -- assumed homogenous!
    -> Rank :: Integer index of current process in the total world size
    -> Local Rank :: Local index on given node in [0, Devices per Node]
"""

import os
import random
from typing import Callable, Optional, Union

import numpy as np
import torch

# === Devices ===


def xpu_is_available() -> bool:
    """Returns True if PyTorch was built with Intel XPU support and an Intel GPU is visible."""
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def get_default_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Returns the device to run inference on.

    An explicit `device` ("cuda", "xpu", "cpu", "xpu:1", ...) is used as given; without one, CUDA if available,
    else Intel XPU, else the CPU.
    """
    if device:
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(f"Device {device} was requested, but CUDA is not available")
        if device.type == "xpu" and not xpu_is_available():
            raise ValueError(f"Device {device} was requested, but no Intel XPU is available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if xpu_is_available():
        return torch.device("xpu:0")
    return torch.device("cpu")


# === Randomness ===


def set_global_seed(seed: int, get_worker_init_fn: bool = False) -> Optional[Callable[[int], None]]:
    """Sets seed for all randomness libraries (mostly random, numpy, torch) and produces a `worker_init_fn`"""
    assert np.iinfo(np.uint32).min < seed < np.iinfo(np.uint32).max, "Seed outside the np.uint32 bounds!"

    # Set Seed as an Environment Variable
    os.environ["EXPERIMENT_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return worker_init_function if get_worker_init_fn else None


def worker_init_function(worker_id: int) -> None:
    """
    Borrowed directly from PyTorch-Lightning; inspired by this issue comment in the PyTorch repo:
        > Ref: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562

    Intuition: You can think of the seed sequence spawn function as a "janky" torch.Generator() or jax.PRNGKey that
    you can run iterative splitting on to get new (predictable) randomness.

    :param worker_id: Identifier for the given worker [0, num_workers) for the Dataloader in question.
    """
    # Get current `rank` (if running distributed) and `process_seed`
    global_rank, process_seed = int(os.environ["LOCAL_RANK"]), torch.initial_seed()

    # Back out the "base" (original) seed - the per-worker seed is set in PyTorch:
    #   > https://pytorch.org/docs/stable/data.html#data-loading-randomness
    base_seed = process_seed - worker_id

    # "Magic" code --> basically creates a seed sequence that mixes different "sources" and seeds every library...
    seed_seq = np.random.SeedSequence([base_seed, worker_id, global_rank])

    # Use 128 bits (4 x 32-bit words) to represent seed --> generate_state(k) produces a `k` element array!
    np.random.seed(seed_seq.generate_state(4))

    # Spawn distinct child sequences for PyTorch (reseed) and stdlib random
    torch_seed_seq, random_seed_seq = seed_seq.spawn(2)

    # Torch Manual seed takes 64 bits (so just specify a dtype of uint64
    torch.manual_seed(torch_seed_seq.generate_state(1, dtype=np.uint64)[0])

    # Use 128 Bits for `random`, but express as integer instead of as an array
    random_seed = (random_seed_seq.generate_state(2, dtype=np.uint64).astype(list) * [1 << 64, 1]).sum()
    random.seed(random_seed)


# === BFloat16 Support ===


def check_bloat16_supported() -> bool:
    try:
        import packaging.version
        import torch.cuda.nccl as nccl
        import torch.distributed as dist

        return (
            (torch.version.cuda is not None)
            and torch.cuda.is_bf16_supported()
            and (packaging.version.parse(torch.version.cuda).release >= (11, 0))
            and dist.is_nccl_available()
            and (nccl.version() >= (2, 10))
        )

    except Exception:
        return False
