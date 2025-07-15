#!/usr/bin/env python3
"""
Run InternVL model on satellite images and compare with ground truth
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer
from internvl_chat.test import load_image

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
    pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
    
    # Define the prompt
    prompt = ("<image>\n"
              "From this aerial image of an urban street scene, identify and trace all visible road markings, "
              "including lane dividers, lane boundaries, bike lanes. For each marking, output a polyline or a sequence "
              "of (x, y) pixel coordinates representing its shape. Only include visible markings painted on the road surface.")
    
    # Generation configuration
    generation_config = dict(
        max_new_tokens=1024,
        do_sample=True,
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
    
    print(f"Comparison saved to: {output_path}")

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
    
    print(f"Overlay saved to: {output_path}")

def main():
    """Main function to run model inference and create comparisons"""
    
    # Directory paths
    satmap_dir = "/home/paperspace/Developer/InternVL/internvl_chat/examples/satmap"
    output_dir = "/home/paperspace/Developer/InternVL/satmap_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model_path = "OpenGVLab/InternVL3-2B"
    print(f"Loading model: {model_path}")
    
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map='auto'
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    
    # Get all image files
    image_files = [f for f in os.listdir(satmap_dir) if f.endswith('.png')]
    image_files.sort()
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for i, img_file in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] Processing: {img_file}")
        
        # File paths
        image_path = os.path.join(satmap_dir, img_file)
        json_path = os.path.join(satmap_dir, img_file.replace('.png', '.json'))
        
        # Load ground truth
        gt_polylines = load_ground_truth_polylines(json_path)
        print(f"  Ground truth: {len(gt_polylines)} polylines")
        
        # Run model inference
        print("  Running model inference...")
        response = run_model_inference(model, tokenizer, image_path)
        
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
        
        print(f"  Model response saved to: {response_path}")
        print(f"  Visualizations created for {base_name}")
    
    print(f"\n✅ All processing complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()