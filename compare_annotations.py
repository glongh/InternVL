#!/usr/bin/env python3
"""
Compare ground truth annotations with model predictions
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from transformers import AutoModel, AutoTokenizer
from internvl_chat.test import load_image

def load_ground_truth(json_path):
    """Load ground truth annotations from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    img_name = list(data.keys())[0]
    lines = []
    for line in data[img_name]['lines']:
        points = np.array(line['points']).astype(np.int32)
        lines.append(points)
    
    return lines, data[img_name]

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
            
            if coords:
                polylines.append(np.array(coords))
    
    return polylines

def visualize_comparison(image_path, ground_truth_lines, predicted_lines, save_path=None):
    """Create side-by-side comparison of ground truth and predictions"""
    
    # Load image
    img = mpimg.imread(image_path)
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
    
    # Original image
    ax1.imshow(img)
    ax1.set_title("Original Image", fontsize=16)
    ax1.axis('off')
    
    # Ground truth
    ax2.imshow(img, alpha=0.7)
    ax2.set_title(f"Ground Truth ({len(ground_truth_lines)} lines)", fontsize=16)
    ax2.axis('off')
    
    for i, line in enumerate(ground_truth_lines):
        ax2.plot(line[:, 0], line[:, 1], 
                 color='lime', 
                 linewidth=2, 
                 alpha=0.8)
    
    # Model predictions
    ax3.imshow(img, alpha=0.7)
    ax3.set_title(f"Model Predictions ({len(predicted_lines)} lines)", fontsize=16)
    ax3.axis('off')
    
    for i, line in enumerate(predicted_lines):
        color = plt.cm.rainbow(i / max(len(predicted_lines), 1))
        ax3.plot(line[:, 0], line[:, 1], 
                 color=color, 
                 linewidth=2, 
                 alpha=0.8,
                 marker='o',
                 markersize=2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison saved to: {save_path}")
    
    plt.show()
    
    return fig

def overlay_comparison(image_path, ground_truth_lines, predicted_lines, save_path=None):
    """Overlay ground truth and predictions on same image"""
    
    # Load image
    img = mpimg.imread(image_path)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    
    # Display image
    ax.imshow(img, alpha=0.6)
    ax.set_title("Ground Truth (green) vs Predictions (red)", fontsize=16)
    ax.axis('off')
    
    # Plot ground truth in green
    for line in ground_truth_lines:
        ax.plot(line[:, 0], line[:, 1], 
                color='lime', 
                linewidth=3, 
                alpha=0.8,
                label='Ground Truth' if line is ground_truth_lines[0] else '')
    
    # Plot predictions in red
    for line in predicted_lines:
        ax.plot(line[:, 0], line[:, 1], 
                color='red', 
                linewidth=2, 
                alpha=0.8,
                linestyle='--',
                label='Prediction' if line is predicted_lines[0] else '')
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    ax.legend(unique_handles, unique_labels, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Overlay saved to: {save_path}")
    
    plt.show()
    
    return fig

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare ground truth with model predictions')
    parser.add_argument('--image', type=str, required=True, help='Path to satellite image')
    parser.add_argument('--gt-json', type=str, required=True, help='Path to ground truth JSON')
    parser.add_argument('--model-path', type=str, help='Path to model checkpoint')
    parser.add_argument('--model-output', type=str, help='Model output string (if already generated)')
    parser.add_argument('--save-comparison', type=str, help='Path to save comparison image')
    parser.add_argument('--save-overlay', type=str, help='Path to save overlay image')
    
    args = parser.parse_args()
    
    # Load ground truth
    print("Loading ground truth annotations...")
    gt_lines, gt_metadata = load_ground_truth(args.gt_json)
    print(f"Loaded {len(gt_lines)} ground truth lines")
    
    # Get model predictions
    if args.model_output:
        # Use provided model output
        predicted_lines = parse_model_predictions(args.model_output)
    elif args.model_path:
        # Generate predictions using model
        print(f"Loading model from {args.model_path}...")
        model = AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map='auto'
        ).eval()
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
        
        # Load image
        pixel_values = load_image(args.image, max_num=12).to(torch.bfloat16).cuda()
        
        # Generate prediction
        prompt = ("<image>\nFrom this aerial image of an urban street scene, identify and trace all visible road markings, "
                 "including lane dividers, lane boundaries, bike lanes. For each marking, output a polyline or a sequence "
                 "of (x, y) pixel coordinates representing its shape. Only include visible markings painted on the road surface.")
        
        generation_config = dict(max_new_tokens=1024, do_sample=True, temperature=0.7)
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)
        
        print(f"Model response:\n{response}")
        predicted_lines = parse_model_predictions(response)
    else:
        print("Please provide either --model-output or --model-path")
        exit(1)
    
    print(f"Parsed {len(predicted_lines)} predicted lines")
    
    # Create visualizations
    if args.save_comparison:
        visualize_comparison(args.image, gt_lines, predicted_lines, args.save_comparison)
    
    if args.save_overlay:
        overlay_comparison(args.image, gt_lines, predicted_lines, args.save_overlay)
    
    # If no save paths provided, show both
    if not args.save_comparison and not args.save_overlay:
        visualize_comparison(args.image, gt_lines, predicted_lines)
        overlay_comparison(args.image, gt_lines, predicted_lines)