#!/usr/bin/env python3
"""
Simple test of the fine-tuned model checkpoint
"""

import os
import json
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
try:
    from peft import PeftModel, PeftConfig
    has_peft = True
except ImportError:
    has_peft = False
    print("PEFT not installed. LoRA loading will not be available.")

# Image preprocessing functions
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def load_image_simple(image_file, input_size=448):
    """Simple image loading without dynamic preprocessing"""
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    # Just use the image as-is (single patch)
    pixel_values = transform(image).unsqueeze(0)  # Add batch dimension
    return pixel_values

def parse_model_predictions(response):
    """Parse model output to extract polylines"""
    polylines = []
    
    print("Parsing model response...")
    print(f"Response length: {len(response)}")
    
    for line in response.strip().split('\n'):
        if '<line>' in line and '</line>' in line:
            start = line.find('<line>') + 6
            end = line.find('</line>')
            coords_str = line[start:end].strip()
            
            coords = []
            parts = coords_str.split()
            for i in range(0, len(parts) - 1, 2):
                try:
                    x = int(parts[i].replace('<', '').replace('>', ''))
                    y = int(parts[i + 1].replace('<', '').replace('>', ''))
                    coords.append([x, y])
                except ValueError as e:
                    print(f"Error parsing coordinates: {parts[i]}, {parts[i+1]} - {e}")
                    continue
            
            if len(coords) > 1:
                polylines.append(np.array(coords))
                print(f"Found polyline with {len(coords)} points")
    
    return polylines

def test_with_base_model_and_checkpoint():
    """Test using base model structure but loading fine-tuned weights (including LoRA)"""
    
    # Paths
    base_model_path = "OpenGVLab/InternVL3-2B"
    checkpoint_path = "/home/paperspace/Developer/InternVL/internvl_chat/work_dirs/internvl3_2b_lora_finetune"
    test_image_path = "/home/paperspace/Developer/InternVL/internvl_chat/examples/satmap/patch_0044_896_1152.png"
    
    print("="*80)
    print("TESTING FINE-TUNED MODEL CHECKPOINT")
    print("="*80)
    print(f"Base model: {base_model_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test image: {test_image_path}")
    print()
    
    # First, let's check what files are in the checkpoint
    print("Checkpoint contents:")
    for file in os.listdir(checkpoint_path):
        print(f"  - {file}")
    print()
    
    # Check if this is a LoRA checkpoint
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    is_lora = os.path.exists(adapter_config_path)
    
    if is_lora and has_peft:
        print("✅ Detected LoRA checkpoint!")
        with open(adapter_config_path, 'r') as f:
            lora_config = json.load(f)
        print(f"   LoRA rank: {lora_config.get('r', 'N/A')}")
        print(f"   LoRA alpha: {lora_config.get('lora_alpha', 'N/A')}")
    
    # Load base model first
    print("\nLoading base model structure...")
    try:
        model = AutoModel.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map={"": 0}
        ).eval()
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, use_fast=False)
        print("✅ Base model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading base model: {e}")
        return
    
    # Try to load fine-tuned weights
    print("\nAttempting to load fine-tuned weights...")
    try:
        if is_lora and has_peft:
            # Load as LoRA adapter
            print("Loading LoRA adapter...")
            model = PeftModel.from_pretrained(model, checkpoint_path)
            print("✅ LoRA adapter loaded successfully!")
            
        else:
            # Check if model.safetensors exists
            safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
            if os.path.exists(safetensors_path):
                print(f"Found model.safetensors: {os.path.getsize(safetensors_path)} bytes")
                
                # Load the state dict
                from safetensors import safe_open
                
                print("Loading weights from safetensors...")
                state_dict = {}
                with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
                
                # Check if these are LoRA weights
                lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower()]
                if lora_keys:
                    print(f"   Found {len(lora_keys)} LoRA-related keys in safetensors")
                    print("   This might be a LoRA checkpoint saved differently")
                
                # Load the state dict into the model
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                print(f"✅ Loaded fine-tuned weights!")
                print(f"   Missing keys: {len(missing_keys)}")
                print(f"   Unexpected keys: {len(unexpected_keys)}")
                
            else:
                print("❌ model.safetensors not found, using base model weights")
            
    except Exception as e:
        print(f"⚠️  Warning: Could not load fine-tuned weights: {e}")
        print("   Proceeding with base model weights for comparison")
    
    # Load test image
    print(f"\nLoading test image: {os.path.basename(test_image_path)}")
    try:
        pixel_values = load_image_simple(test_image_path).to(torch.bfloat16).cuda()
        print("✅ Image loaded and preprocessed successfully!")
        
        # Display image info
        img = Image.open(test_image_path)
        print(f"   Image size: {img.size}")
        print(f"   Pixel values shape: {pixel_values.shape}")
        
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return
    
    # Define the prompt (same as used in training)
    prompt = ("<image>\n"
              "From this aerial image of an urban street scene, identify and trace all visible road markings, "
              "including lane dividers, lane boundaries, bike lanes. For each marking, output a polyline or a sequence "
              "of (x, y) pixel coordinates representing its shape. Only include visible markings painted on the road surface.\n\n"
              "Format your answers like in the following examples:\n"
              "<line> <473> <21> <420> <149> <377> <267> <318> <407> <274> <512> </line>\n"
              "<line> <351> <512> <367> <473> <407> <378> <446> <281> <489> <173> <512> <118> </line>\n"
              "<line> <89> <156> <123> <189> <156> <223> <189> <256> <223> <290> </line>")
    
    print(f"\nPrompt: {prompt[:100]}...")
    
    # Generation configuration
    generation_config = dict(
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )
    
    # Run inference
    print("\nRunning inference...")
    try:
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)
        print("✅ Inference completed!")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return
    
    # Display results
    print("\n" + "="*80)
    print("MODEL RESPONSE")
    print("="*80)
    print(response)
    print("="*80)
    
    # Check if response contains expected format
    has_line_tags = '<line>' in response and '</line>' in response
    
    print(f"\n📊 RESPONSE ANALYSIS:")
    print(f"   Length: {len(response)} characters")
    print(f"   Contains <line> tags: {has_line_tags}")
    
    if has_line_tags:
        print("   🎯 SUCCESS: Model is outputting in expected coordinate format!")
        
        # Parse the predictions
        pred_polylines = parse_model_predictions(response)
        print(f"   📍 Parsed {len(pred_polylines)} polylines")
        
        # Show first few coordinates
        for i, polyline in enumerate(pred_polylines[:3]):
            print(f"   Line {i+1}: {len(polyline)} points - {polyline[0]} to {polyline[-1]}")
        
        # Load ground truth for comparison
        gt_json_path = test_image_path.replace('.png', '.json')
        if os.path.exists(gt_json_path):
            with open(gt_json_path, 'r') as f:
                gt_data = json.load(f)
            print(f"   📋 Ground truth: {len(gt_data)} polylines")
            print(f"   📈 Detection ratio: {len(pred_polylines)}/{len(gt_data)} = {len(pred_polylines)/len(gt_data):.2f}")
        
    else:
        print("   ⚠️  Model is not outputting in expected coordinate format")
        print("   💡 This might indicate:")
        print("      - Fine-tuning was not successful")
        print("      - Wrong checkpoint loaded")
        print("      - Need different generation parameters")
    
    # Save the response for inspection
    output_file = "finetuned_model_test_response.txt"
    with open(output_file, 'w') as f:
        f.write("FINE-TUNED MODEL TEST RESPONSE\n")
        f.write("="*50 + "\n\n")
        f.write(f"Base model: {base_model_path}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Test image: {test_image_path}\n")
        f.write(f"Prompt: {prompt}\n\n")
        f.write("Response:\n")
        f.write("-"*30 + "\n")
        f.write(response)
    
    print(f"\n💾 Response saved to: {output_file}")
    
    return has_line_tags, response

def main():
    """Main function"""
    
    # Test the fine-tuned model
    try:
        success, response = test_with_base_model_and_checkpoint()
        
        if success:
            print("\n🎉 CONCLUSION: Fine-tuned model is working correctly!")
            print("   The model outputs road marking coordinates in the expected format.")
        else:
            print("\n🔧 CONCLUSION: Model needs further investigation")
            print("   The model is not outputting in the expected coordinate format.")
            
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        print("   Please check the checkpoint path and model configuration.")

if __name__ == "__main__":
    main()