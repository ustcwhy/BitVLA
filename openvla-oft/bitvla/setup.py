from setuptools import setup, find_packages

setup(
    name="bitvla", 
    version="0.1",            
    packages=find_packages(),  
    py_modules=["configuration_bit_vla"],  # imported as a top-level module by model/bitvla_for_action_prediction.py
    author="lxsy_chuyan_xiong",        
    author_email="lxsy31084573@163.com",  
    
    package = find_packages(),  
)
