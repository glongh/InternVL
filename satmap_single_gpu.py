#!/usr/bin/env python3
"""
Run InternVL model on satellite images and compare with ground truth - Single GPU version
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

def load_ground_truth_polylines(json_path):
    """Load ground truth polylines from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert to numpy arrays
    polylines = []
    for line in data:
        if len(line) > 1:  # Only keep lines with multiple points
            polylines.append(np.array(line))
    
    return polylines

def parse_model_predictions(response):
    """Parse model output to extract polylines"""
    polylines = []
    
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
                except:
                    continue
            
            if len(coords) > 1:  # Only keep lines with multiple points
                polylines.append(np.array(coords))
    
    return polylines

def run_model_inference(model, tokenizer, image_path):
    """Run model inference on a single image"""
    
    # Load and preprocess image
    pixel_values = load_image(image_path, max_num=6).to(torch.bfloat16).cuda()  # Reduced max_num for memory
    
    # Define the prompt
    prompt = ("<image>\n"
              "From this aerial image of an urban street scene, identify and trace all visible road markings, "
              "including lane dividers, lane boundaries, bike lanes. For each marking, output a polyline or a sequence "
              "of (x, y) pixel coordinates representing its shape. Only include visible markings painted on the road surface.")
    
    # Generation configuration
    generation_config = dict(
        max_new_tokens=512,  # Reduced for memory
        do_sample=False,     # Deterministic generation
        temperature=0.7,
        top_p=0.95
    )
    
    # Get model response
    response = model.chat(tokenizer, pixel_values, prompt, generation_config)
    
    return response

def create_comparison_visualization(image_path, gt_polylines, pred_polylines, output_path):
    """Create side-by-side comparison visualization"""
    
    # Load image
    img = mpimg.imread(image_path)
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    ax1.imshow(img)
    ax1.set_title("Original Image", fontsize=14)
    ax1.axis('off')
    
    # Ground truth
    ax2.imshow(img, alpha=0.7)
    ax2.set_title(f"Ground Truth ({len(gt_polylines)} lines)", fontsize=14)
    ax2.axis('off')
    
    for i, line in enumerate(gt_polylines):
        if len(line) > 1:
            ax2.plot(line[:, 0], line[:, 1], 
                     color='lime', 
                     linewidth=2, 
                     alpha=0.8,
                     marker='o',
                     markersize=1)
    
    # Model predictions
    ax3.imshow(img, alpha=0.7)
    ax3.set_title(f"Model Predictions ({len(pred_polylines)} lines)", fontsize=14)
    ax3.axis('off')
    
    for i, line in enumerate(pred_polylines):
        if len(line) > 1:
            color = plt.cm.rainbow(i / max(len(pred_polylines), 1))
            ax3.plot(line[:, 0], line[:, 1], 
                     color=color, 
                     linewidth=2, 
                     alpha=0.8,
                     marker='o',
                     markersize=1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

def create_overlay_visualization(image_path, gt_polylines, pred_polylines, output_path):
    """Create overlay visualization with ground truth and predictions"""
    
    # Load image
    img = mpimg.imread(image_path)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Display image
    ax.imshow(img, alpha=0.8)
    ax.set_title("Ground Truth (Green) vs Predictions (Red)", fontsize=16)
    ax.axis('off')
    
    # Plot ground truth in green
    for i, line in enumerate(gt_polylines):
        if len(line) > 1:
            ax.plot(line[:, 0], line[:, 1], 
                    color='lime', 
                    linewidth=3, 
                    alpha=0.9,
                    label='Ground Truth' if i == 0 else '')
    
    # Plot predictions in red
    for i, line in enumerate(pred_polylines):
        if len(line) > 1:
            ax.plot(line[:, 0], line[:, 1], 
                    color='red', 
                    linewidth=2, 
                    alpha=0.8,
                    linestyle='--',
                    label='Prediction' if i == 0 else '')
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique_labels = []
        unique_handles = []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
        ax.legend(unique_handles, unique_labels, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run model inference and create comparisons"""
    
    # Directory paths
    satmap_dir = "/home/paperspace/Developer/InternVL/internvl_chat/examples/satmap"
    output_dir = "/home/paperspace/Developer/InternVL/satmap_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer - Force single GPU
    model_path = "OpenGVLab/InternVL3-2B"
    print(f"Loading model: {model_path}")
    
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map={"": 0}  # Force everything to GPU 0
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    
    # Get all image files
    image_files = [f for f in os.listdir(satmap_dir) if f.endswith('.png')]
    image_files.sort()
    
    print(f"Found {len(image_files)} images to process")
    
    # Process first few images for demo
    for i, img_file in enumerate(image_files[:3]):  # Process first 3 images
        print(f"\n[{i+1}/3] Processing: {img_file}")
        
        # File paths
        image_path = os.path.join(satmap_dir, img_file)
        json_path = os.path.join(satmap_dir, img_file.replace('.png', '.json'))
        
        # Load ground truth
        gt_polylines = load_ground_truth_polylines(json_path)
        print(f"  Ground truth: {len(gt_polylines)} polylines")
        
        # Run model inference
        print("  Running model inference...")
        try:
            response = run_model_inference(model, tokenizer, image_path)
            print(f"  Model response: {response[:100]}...")
            
            # Parse predictions
            pred_polylines = parse_model_predictions(response)
            print(f"  Predictions: {len(pred_polylines)} polylines")
            
            # Create output filenames
            base_name = img_file.replace('.png', '')
            comparison_path = os.path.join(output_dir, f"{base_name}_comparison.png")
            overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
            
            # Create visualizations
            create_comparison_visualization(image_path, gt_polylines, pred_polylines, comparison_path)
            create_overlay_visualization(image_path, gt_polylines, pred_polylines, overlay_path)
            
            # Save model response
            response_path = os.path.join(output_dir, f"{base_name}_response.txt")
            with open(response_path, 'w') as f:
                f.write(response)
            
            print(f"  ✅ Completed {base_name}")
            
        except Exception as e:
            print(f"  ❌ Error processing {img_file}: {e}")
            continue
    
    print(f"\n✅ Processing complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()