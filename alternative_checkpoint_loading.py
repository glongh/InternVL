#!/usr/bin/env python3
"""
Alternative method to load the fine-tuned checkpoint
"""

import torch
from transformers import AutoModel, AutoTokenizer
from safetensors import safe_open
import os

def load_checkpoint_manually():
    """Try to load checkpoint manually"""
    
    base_model_path = "OpenGVLab/InternVL3-2B"
    checkpoint_path = "/home/paperspace/Developer/InternVL/internvl_chat/work_dirs/checkpoint-10200"
    
    print("Loading base model...")
    model = AutoModel.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map={"": 0}
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, use_fast=False)
    
    print("Loading checkpoint weights...")
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    
    try:
        # Load state dict from safetensors
        state_dict = {}
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        
        # Load into model
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        print(f"Loaded checkpoint successfully!")
        print(f"Missing keys: {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return None, None

if __name__ == "__main__":
    model, tokenizer = load_checkpoint_manually()
    if model is not None:
        print("✅ Model loaded successfully!")
        print("   You can now use this model for inference")
    else:
        print("❌ Failed to load model")
