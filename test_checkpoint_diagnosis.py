#!/usr/bin/env python3
"""
Diagnosis script for the fine-tuned model checkpoint
"""

import os
import json
import torch
from safetensors import safe_open

def diagnose_checkpoint():
    """Diagnose the checkpoint to understand its structure"""
    
    checkpoint_path = "/home/paperspace/Developer/InternVL/internvl_chat/work_dirs/checkpoint-10200"
    
    print("="*80)
    print("CHECKPOINT DIAGNOSIS")
    print("="*80)
    print(f"Checkpoint path: {checkpoint_path}")
    print()
    
    # Check all files in checkpoint
    print("📁 Files in checkpoint:")
    for file in sorted(os.listdir(checkpoint_path)):
        file_path = os.path.join(checkpoint_path, file)
        size = os.path.getsize(file_path)
        print(f"   {file:<25} {size:>10} bytes")
    print()
    
    # Check config.json
    config_path = os.path.join(checkpoint_path, "config.json")
    if os.path.exists(config_path):
        print("📄 config.json contents:")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        important_keys = ['model_type', 'architectures', 'num_hidden_layers', 'vocab_size']
        for key in important_keys:
            if key in config:
                print(f"   {key}: {config[key]}")
        print()
    
    # Check trainer_state.json
    trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
    if os.path.exists(trainer_state_path):
        print("📊 trainer_state.json contents:")
        with open(trainer_state_path, 'r') as f:
            trainer_state = json.load(f)
        
        important_keys = ['epoch', 'global_step', 'train_loss', 'eval_loss']
        for key in important_keys:
            if key in trainer_state:
                print(f"   {key}: {trainer_state[key]}")
        
        # Check if training completed
        if 'log_history' in trainer_state:
            print(f"   Training steps: {len(trainer_state['log_history'])}")
            if trainer_state['log_history']:
                last_log = trainer_state['log_history'][-1]
                print(f"   Last log: {last_log}")
        print()
    
    # Check model.safetensors
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    if os.path.exists(safetensors_path):
        print("🔍 model.safetensors analysis:")
        try:
            with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                print(f"   Number of parameters: {len(keys)}")
                print(f"   First 10 parameter names:")
                for key in keys[:10]:
                    tensor = f.get_tensor(key)
                    print(f"     {key:<50} {tensor.shape}")
                
                # Check if it has language model parameters
                language_params = [k for k in keys if 'language_model' in k]
                vision_params = [k for k in keys if 'vision_model' in k]
                
                print(f"   Language model parameters: {len(language_params)}")
                print(f"   Vision model parameters: {len(vision_params)}")
                
        except Exception as e:
            print(f"   ❌ Error reading safetensors: {e}")
    
    print()
    
    # Try to understand the training setup
    training_args_path = os.path.join(checkpoint_path, "training_args.bin")
    if os.path.exists(training_args_path):
        print("⚙️ Training arguments:")
        try:
            training_args = torch.load(training_args_path, map_location='cpu')
            important_args = ['learning_rate', 'num_train_epochs', 'per_device_train_batch_size', 'gradient_accumulation_steps']
            for arg in important_args:
                if hasattr(training_args, arg):
                    print(f"   {arg}: {getattr(training_args, arg)}")
        except Exception as e:
            print(f"   ❌ Error reading training args: {e}")
    
    print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS:")
    print("-"*40)
    
    # Check if this is a proper checkpoint
    if os.path.exists(safetensors_path) and os.path.exists(config_path):
        print("✅ Checkpoint structure looks correct")
        print("   - Has model.safetensors (model weights)")
        print("   - Has config.json (model configuration)")
        print("   - Has tokenizer files")
        
        # Check if training completed
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, 'r') as f:
                trainer_state = json.load(f)
            
            if 'epoch' in trainer_state and trainer_state['epoch'] > 0:
                print(f"✅ Training completed {trainer_state['epoch']} epochs")
            else:
                print("⚠️  Training might not have completed successfully")
        
        print("\n   To test the model:")
        print("   1. Use the same base model architecture")
        print("   2. Load the checkpoint weights manually")
        print("   3. Test with the exact same prompt format used in training")
        print("   4. Check generation parameters (temperature, top_p, etc.)")
        
    else:
        print("❌ Checkpoint structure is incomplete")
        print("   Missing essential files for model loading")

def create_alternative_loading_script():
    """Create a script to try alternative loading methods"""
    
    script_content = '''#!/usr/bin/env python3
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
'''
    
    with open("alternative_checkpoint_loading.py", "w") as f:
        f.write(script_content)
    
    print("📄 Created alternative loading script: alternative_checkpoint_loading.py")

if __name__ == "__main__":
    diagnose_checkpoint()
    create_alternative_loading_script()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("The checkpoint exists but may have issues with the safetensors format.")
    print("This could be due to:")
    print("1. Incomplete training/saving process")
    print("2. Safetensors version compatibility")
    print("3. Checkpoint corruption")
    print()
    print("Next steps:")
    print("1. Check if training completed successfully")
    print("2. Try alternative loading methods")
    print("3. Re-save the checkpoint if needed")
    print("4. Verify the model outputs the expected format")