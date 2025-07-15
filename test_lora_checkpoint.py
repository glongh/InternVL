#!/usr/bin/env python3
"""
Test loading LoRA checkpoint for InternVL fine-tuned model
"""

import os
import json
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel, PeftConfig

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

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
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

def test_lora_checkpoint():
    """Test loading a LoRA checkpoint"""
    
    # Paths
    base_model_path = "OpenGVLab/InternVL3-2B"
    checkpoint_path = "/home/paperspace/Developer/InternVL/internvl_chat/work_dirs/checkpoint-10200"
    test_image_path = "/home/paperspace/Developer/InternVL/internvl_chat/examples/satmap/patch_0044_896_1152.png"
    
    print("="*80)
    print("TESTING LORA CHECKPOINT LOADING")
    print("="*80)
    print(f"Base model: {base_model_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test image: {test_image_path}")
    print()
    
    # First check if this is a LoRA checkpoint
    print("Checking checkpoint type...")
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    is_lora = os.path.exists(adapter_config_path)
    
    if is_lora:
        print("✅ Found adapter_config.json - This is a LoRA checkpoint!")
        
        # Load LoRA config
        with open(adapter_config_path, 'r') as f:
            lora_config = json.load(f)
        print(f"   LoRA rank: {lora_config.get('r', 'N/A')}")
        print(f"   LoRA alpha: {lora_config.get('lora_alpha', 'N/A')}")
        print(f"   Target modules: {lora_config.get('target_modules', 'N/A')}")
        
        # Load base model
        print("\nLoading base model...")
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
        
        # Load LoRA weights
        print("\nLoading LoRA adapter...")
        try:
            model = PeftModel.from_pretrained(model, checkpoint_path)
            print("✅ LoRA adapter loaded successfully!")
            
            # Optional: merge LoRA weights for faster inference
            # model = model.merge_and_unload()
            # print("✅ LoRA weights merged into base model!")
            
        except Exception as e:
            print(f"❌ Error loading LoRA adapter: {e}")
            print("   Trying alternative loading method...")
            
    else:
        print("ℹ️  No adapter_config.json found - This appears to be a full model checkpoint")
        print("   Loading as a regular checkpoint...")
        
        # Try to load as a full model
        print("\nLoading model from checkpoint...")
        try:
            # First try to load directly from checkpoint
            model = AutoModel.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map={"": 0}
            ).eval()
            
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True, use_fast=False)
            print("✅ Model loaded successfully from checkpoint!")
            
        except Exception as e:
            print(f"⚠️  Could not load directly from checkpoint: {e}")
            print("   Trying to load base model and then weights...")
            
            # Load base model and then try to load weights
            try:
                model = AutoModel.from_pretrained(
                    base_model_path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    device_map={"": 0}
                ).eval()
                
                tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, use_fast=False)
                
                # Load the state dict
                safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
                if os.path.exists(safetensors_path):
                    from safetensors import safe_open
                    
                    state_dict = {}
                    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            state_dict[key] = f.get_tensor(key)
                    
                    # Check if these are LoRA weights by examining the keys
                    lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower()]
                    if lora_keys:
                        print(f"   Found {len(lora_keys)} LoRA-related keys in checkpoint")
                        print("   This appears to be a LoRA checkpoint saved as safetensors")
                    
                    # Load the state dict
                    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                    print(f"✅ Loaded checkpoint weights!")
                    print(f"   Missing keys: {len(missing_keys)}")
                    print(f"   Unexpected keys: {len(unexpected_keys)}")
                    
                    if unexpected_keys:
                        print("   Sample unexpected keys:", unexpected_keys[:5])
                
            except Exception as e2:
                print(f"❌ Failed to load model: {e2}")
                return
    
    # Load test image
    print(f"\nLoading test image: {os.path.basename(test_image_path)}")
    try:
        pixel_values = load_image(test_image_path, max_num=6).to(torch.bfloat16).cuda()
        print("✅ Image loaded and preprocessed successfully!")
        
        # Display image info
        img = Image.open(test_image_path)
        print(f"   Image size: {img.size}")
        print(f"   Pixel values shape: {pixel_values.shape}")
        
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return
    
    # Define the prompt
    prompt = ("<image>\n"
              "From this aerial image of an urban street scene, identify and trace all visible road markings, "
              "including lane dividers, lane boundaries, bike lanes. For each marking, output a polyline or a sequence "
              "of (x, y) pixel coordinates representing its shape. Only include visible markings painted on the road surface.")
    
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
        
    else:
        print("   ⚠️  Model is not outputting in expected coordinate format")
        print("   💡 Possible reasons:")
        print("      - LoRA weights not properly loaded")
        print("      - Model needs more training")
        print("      - Generation parameters need adjustment")
    
    return has_line_tags, response

def main():
    """Main function"""
    
    # Test the checkpoint
    try:
        success, response = test_lora_checkpoint()
        
        if success:
            print("\n🎉 CONCLUSION: Model checkpoint loaded and working correctly!")
        else:
            print("\n🔧 CONCLUSION: Checkpoint loading needs investigation")
            
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()