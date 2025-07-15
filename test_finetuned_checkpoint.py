#!/usr/bin/env python3
"""
Test the fine-tuned model checkpoint to see if it responds in the expected format
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

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

def test_finetuned_model():
    """Test the fine-tuned model checkpoint"""
    
    # Model paths
    checkpoint_path = "/home/paperspace/Developer/InternVL/internvl_chat/work_dirs/checkpoint-10200"
    test_image_path = "/home/paperspace/Developer/InternVL/internvl_chat/examples/satmap/patch_0044_896_1152.png"
    
    print("="*80)
    print("TESTING FINE-TUNED MODEL CHECKPOINT")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test image: {test_image_path}")
    print()
    
    # Load model and tokenizer
    print("Loading fine-tuned model...")
    try:
        model = AutoModel.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map={"": 0}  # Force single GPU
        ).eval()
        
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True, use_fast=False)
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
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
    
    # Define the prompt (same as used in training)
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
        print("   ⚠️  WARNING: Model is not outputting in expected coordinate format")
        print("   💡 This might indicate the model needs more training or different parameters")
    
    # Save the response for inspection
    output_file = "finetuned_model_test_response.txt"
    with open(output_file, 'w') as f:
        f.write("FINE-TUNED MODEL TEST RESPONSE\n")
        f.write("="*50 + "\n\n")
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
    success, response = test_finetuned_model()
    
    if success:
        print("\n🎉 CONCLUSION: Fine-tuned model is working correctly!")
        print("   The model outputs road marking coordinates in the expected format.")
    else:
        print("\n🔧 CONCLUSION: Model needs further investigation")
        print("   The model is not outputting in the expected coordinate format.")
        print("   Consider checking:")
        print("   - Training parameters")
        print("   - Model convergence")
        print("   - Data format consistency")

if __name__ == "__main__":
    main()