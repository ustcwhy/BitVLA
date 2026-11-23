import torch
from typing import Any, Dict, List, Optional
import numpy as np
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoProcessor,
    AutoModelForVision2Seq,
    SiglipImageProcessor,
    LlavaProcessor,
)
import sys
sys.path.append("../..")
from bitvla import Bitvla_Config,BitVLAForActionPrediction
from transformers.image_utils import get_image_size,to_numpy_array
import copy
from bitvla.constants import (
    BITNET_DEFAULT_IM_END_TOKEN,
    BITNET_DEFAULT_IMAGE_TOKEN,
)
from experiments.robot.openvla_utils import (
    model_is_on_hf_hub,
    update_auto_map,
    check_model_logic_mismatch,
    _load_dataset_stats,
    prepare_images_for_vla,
    normalize_proprio,
)
from prismatic.util.torch_utils import get_default_device

DEVICE = get_default_device()
BITNET_VLA_IMAGE_SIZE = 224
from bitvla.dataset.bitvla_transform import llava_to_openai



def get_bitnet_vla(cfg: Any) -> torch.nn.Module:
    """
    Load and initialize the VLA model from checkpoint.

    Args:
        cfg: Configuration object

    Returns:
        torch.nn.Module: The initialized VLA model
    """
    print("Instantiating pretrained VLA policy...")

    # If loading a locally stored pretrained checkpoint, check whether config or model files
    # need to be synced so that any changes the user makes to the VLA modeling code will
    # actually go into effect
    # If loading a pretrained checkpoint from Hugging Face Hub, we just assume that the policy
    # will be used as is, with its original modeling logic
    if not model_is_on_hf_hub(cfg.pretrained_checkpoint):
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("bitvla", Bitvla_Config)
        AutoImageProcessor.register(Bitvla_Config, SiglipImageProcessor)
        AutoProcessor.register(Bitvla_Config, LlavaProcessor)
        AutoModelForVision2Seq.register(Bitvla_Config, BitVLAForActionPrediction)

        # Update config.json and sync model files
        update_auto_map(cfg.pretrained_checkpoint)
        check_model_logic_mismatch(
            cfg.pretrained_checkpoint,
            curr_files = {"bitvla_for_action_prediction.py":None,"configuration_bit_vla.py":None},
            where_to_find_files_cur_codebase="./bitvla")

    # Load the model
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        trust_remote_code=True,
    )

    vla.eval()

    # Move model to device if not using quantization
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    # Load dataset stats for action normalization
    _load_dataset_stats(vla, cfg.pretrained_checkpoint)

    return vla

def get_bitnet_vla_action(
    cfg: Any,
    vla: torch.nn.Module,
    processor: Any,
    obs: Dict[str, Any],
    task_label: str,
    action_head: Optional[torch.nn.Module] = None,
    proprio_projector: Optional[torch.nn.Module] = None,
    noisy_action_projector: Optional[torch.nn.Module] = None,
    use_film: bool = False,
) -> List[np.ndarray]:
    """
    Generate action predictions with the VLA policy.

    Args:
        cfg: Configuration object with parameters
        vla: The VLA model
        processor: Model processor for inputs
        obs: Observation dictionary(keys: "full_image", "wrist_image", "state")
        task_label: Text description of the task
        action_head: Optional action head for continuous actions
        proprio_projector: Optional proprioception projector
        noisy_action_projector: Optional noisy action projector for diffusion
        use_film: Whether to use FiLM

    Returns:
        List[np.ndarray]: Predicted actions
    """
    with torch.inference_mode():
        
        # Collect all input images
        all_images = [obs["full_image"]]
        if cfg.num_images_in_input > 1:
            all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])

        # Process images
        all_images = prepare_images_for_vla(all_images, cfg, image_size=BITNET_VLA_IMAGE_SIZE)

        # source-like prompt
        sources = {
            "image": all_images,
            "conversations":[
                {
                    "from": "human",
                    "value": "<image>\n"*len(all_images)+"<proprio_pad>"+f"What action should the robot take to {task_label.lower()}?"
                },
                {
                    "from":"gpt",
                    "value": ""
                }
            ]
        }
        pixel_values = [processor.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0] for image in all_images]
        patch_size = processor.patch_size
        num_additional_image_tokens = processor.num_additional_image_tokens
        num_image_tokens = []
        for img in pixel_values:
            height, width = get_image_size(to_numpy_array(img))
            num_image_tokens.append((height // patch_size) * (width // patch_size))
        sources = copy.deepcopy(llava_to_openai(sources['conversations']))
        
        prompt = sources[0]["content"]
        x = [{"role":"user","content":prompt}]
        input_str = processor.tokenizer.apply_chat_template(x,tokenize=False, add_generation_prompt=True)
        # Replace DEFAULT_IMAGE_TOKEN with the appropriate number of tokens
        token_index = 0
        placeholder_token = "<TEMP_IMAGE_TOKEN>"
        input_str = input_str.replace(BITNET_DEFAULT_IMAGE_TOKEN, placeholder_token)
        while placeholder_token in input_str and token_index < len(num_image_tokens):
            input_str = input_str.replace(
                placeholder_token, 
                BITNET_DEFAULT_IMAGE_TOKEN * num_image_tokens[token_index], 
                1
            )
            token_index += 1
        prompt = input_str
        input_ids = processor.tokenizer(prompt, add_special_tokens=True).input_ids
        input_ids = torch.tensor(input_ids)
        # (seq_len) - > (1, seq_len)
        input_ids = input_ids.unsqueeze(0)
        # The attention mask is an input_ids tensor filled with ones
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        pixel_values = torch.stack(pixel_values, dim=0) # (num_image,c,h,w)
        # num_image,c,h,w -> 1,num_image,c,h,w
        pixel_values = pixel_values.unsqueeze(0) # (1,num_image,c,h,w)
        inputs = dict(
            input_ids=input_ids.to(DEVICE, dtype=torch.long),
            attention_mask=attention_mask.to(DEVICE),
            pixel_values=pixel_values.to(DEVICE, dtype=torch.bfloat16),
        )

        # Process proprioception data if used
        proprio = None
        if cfg.use_proprio:
            proprio = obs["state"]
            proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
            obs["state"] = normalize_proprio(proprio, proprio_norm_stats)
            proprio = obs["state"]

        # Generate action
        if action_head is None:
            # Standard VLA output (single-image inputs, discrete actions)
            action, _ = vla.predict_action(**inputs, unnorm_key=cfg.unnorm_key, do_sample=False)
        else:
            # Custom action head for continuous actions
            action, _ = vla.predict_action(
                **inputs,
                unnorm_key=cfg.unnorm_key,
                do_sample=False,
                proprio=proprio,
                proprio_projector=proprio_projector,
                action_head=action_head,
            )

    # Extract subset of actions for open loop steps
    return [action[i] for i in range(min(len(action), cfg.num_open_loop_steps))]