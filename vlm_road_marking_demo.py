#!/usr/bin/env python3
"""
Comprehensive demo showing how InternVL VLM works for road marking detection
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from internvl_chat.test import load_image, build_transform, dynamic_preprocess

def explain_vlm_pipeline():
    """Explain how the VLM processes images and generates road marking coordinates"""
    
    print("="*80)
    print("InternVL Vision-Language Model (VLM) Pipeline for Road Marking Detection")
    print("="*80)
    print()
    
    print("1. IMAGE PREPROCESSING")
    print("-"*40)
    print("   - Input: High-resolution satellite/aerial image (e.g., 512x512 or 4096x4096)")
    print("   - Dynamic preprocessing: Image is split into patches/tiles")
    print("   - Each patch is resized to model input size (typically 448x448)")
    print("   - Normalization with ImageNet statistics")
    print("   - Optional: Thumbnail image added for global context")
    print()
    
    print("2. VISION ENCODER (ViT)")
    print("-"*40)
    print("   - Vision Transformer processes image patches")
    print("   - Extracts visual features from each patch")
    print("   - Creates spatial representations of the image")
    print("   - Output: Visual embeddings/tokens")
    print()
    
    print("3. MULTIMODAL FUSION")
    print("-"*40)
    print("   - Visual tokens are projected to language model dimension")
    print("   - Text prompt is tokenized and embedded")
    print("   - Visual and text embeddings are combined")
    print("   - Special <image> token indicates where visual info is inserted")
    print()
    
    print("4. LANGUAGE MODEL GENERATION")
    print("-"*40)
    print("   - LLM processes combined visual+text input")
    print("   - Generates text output token by token")
    print("   - For road markings: outputs structured coordinate format")
    print("   - Format: <line> <x1> <y1> <x2> <y2> ... </line>")
    print()
    
    print("5. OUTPUT PARSING")
    print("-"*40)
    print("   - Generated text is parsed to extract coordinates")
    print("   - Each <line> represents one road marking/polyline")
    print("   - Coordinates are in pixel space of original image")
    print("   - Can be visualized directly on the input image")
    print()

def visualize_image_preprocessing(image_path, max_num=12):
    """Show how an image is preprocessed for the VLM"""
    
    # Load original image
    image = Image.open(image_path).convert('RGB')
    orig_width, orig_height = image.size
    
    # Apply dynamic preprocessing
    processed_images = dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=max_num)
    
    # Create visualization
    n_patches = len(processed_images)
    cols = int(np.ceil(np.sqrt(n_patches)))
    rows = int(np.ceil(n_patches / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    fig.suptitle(f'Image Preprocessing: {orig_width}x{orig_height} → {n_patches} patches of 448x448', fontsize=16)
    
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, patch in enumerate(processed_images):
        if i < len(axes):
            axes[i].imshow(patch)
            if i == len(processed_images) - 1 and len(processed_images) > 1:
                axes[i].set_title('Thumbnail (global context)', fontsize=10)
            else:
                axes[i].set_title(f'Patch {i+1}', fontsize=10)
            axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(len(processed_images), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def demonstrate_vlm_inference(model_path="OpenGVLab/InternVL3-2B", image_path=None):
    """Demonstrate the full VLM inference pipeline"""
    
    print("\n" + "="*80)
    print("VLM INFERENCE DEMONSTRATION")
    print("="*80)
    
    # Default to demo image if not provided
    if image_path is None:
        image_path = "./internvl_chat/OpensatMap-demo/roundabout5_-1_-1_sat.png"
    
    # The prompt that instructs the model what to do
    prompt = ("<image>\n"
              "From this aerial image of an urban street scene, identify and trace all visible road markings, "
              "including lane dividers, lane boundaries, bike lanes. For each marking, output a polyline or a sequence "
              "of (x, y) pixel coordinates representing its shape. Only include visible markings painted on the road surface.")
    
    print(f"\n1. Loading model: {model_path}")
    print("-"*40)
    
    # Simulate model loading (in real scenario, you'd load the actual model)
    print("   - Loading vision encoder (ViT)")
    print("   - Loading language model")
    print("   - Loading multimodal projector")
    print("   - Model ready for inference")
    
    print(f"\n2. Processing image: {image_path}")
    print("-"*40)
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    print(f"   - Original size: {image.size}")
    
    # Show preprocessing
    transform = build_transform(input_size=448)
    images = dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=12)
    print(f"   - Split into {len(images)} patches")
    print("   - Each patch normalized and transformed")
    
    print("\n3. Prompt Engineering")
    print("-"*40)
    print("   Prompt structure:")
    print("   - <image> token: Indicates where visual features are inserted")
    print("   - Task description: Clear instructions for road marking detection")
    print("   - Output format: Specifies coordinate sequence format")
    
    print("\n4. Model Generation Process")
    print("-"*40)
    print("   The model:")
    print("   a) Encodes the image patches into visual features")
    print("   b) Combines visual features with text prompt")
    print("   c) Generates output token by token")
    print("   d) Produces structured output in the trained format")
    
    # Example output
    example_output = """<line> <1365> <1920> <1346> <1945> <1319> <1970> <1265> <2013> <1227> <2032> </line>
<line> <2048> <1536> <2100> <1550> <2150> <1565> <2200> <1580> <2250> <1595> </line>
<line> <512> <2048> <525> <2100> <538> <2150> <551> <2200> <564> <2250> </line>"""
    
    print("\n5. Example Model Output:")
    print("-"*40)
    print(example_output)
    
    print("\n6. Parsing Output to Coordinates:")
    print("-"*40)
    
    # Parse the example
    lines = []
    for line in example_output.strip().split('\n'):
        if '<line>' in line:
            start = line.find('<line>') + 6
            end = line.find('</line>')
            coords_str = line[start:end].strip()
            coords = []
            parts = coords_str.split()
            for i in range(0, len(parts)-1, 2):
                x = int(parts[i].replace('<', '').replace('>', ''))
                y = int(parts[i+1].replace('<', '').replace('>', ''))
                coords.append((x, y))
            lines.append(coords)
            print(f"   Line {len(lines)}: {len(coords)} points - {coords[:2]}...")
    
    return lines

def visualize_attention_concept():
    """Visualize how the VLM attends to different parts of the image"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Create a simple road scene
    ax1.set_xlim(0, 512)
    ax1.set_ylim(512, 0)  # Invert y-axis to match image coordinates
    ax1.set_aspect('equal')
    
    # Draw roads
    road_color = '#808080'
    ax1.add_patch(patches.Rectangle((150, 0), 212, 512, facecolor=road_color))
    ax1.add_patch(patches.Rectangle((0, 200), 512, 112, facecolor=road_color))
    
    # Draw road markings
    marking_color = 'white'
    # Vertical lane lines
    ax1.plot([206, 206], [0, 200], color=marking_color, linewidth=3)
    ax1.plot([206, 206], [312, 512], color=marking_color, linewidth=3)
    ax1.plot([306, 306], [0, 200], color=marking_color, linewidth=3)
    ax1.plot([306, 306], [312, 512], color=marking_color, linewidth=3)
    
    # Horizontal lane lines
    ax1.plot([0, 150], [256, 256], color=marking_color, linewidth=3)
    ax1.plot([362, 512], [256, 256], color=marking_color, linewidth=3)
    
    # Center dashed line
    for y in range(0, 200, 30):
        ax1.plot([256, 256], [y, y+15], color='yellow', linewidth=3)
    for y in range(312, 512, 30):
        ax1.plot([256, 256], [y, y+15], color='yellow', linewidth=3)
    
    ax1.set_title("Input: Aerial View of Road", fontsize=14)
    ax1.set_xlabel("X (pixels)")
    ax1.set_ylabel("Y (pixels)")
    ax1.grid(True, alpha=0.3)
    
    # Visualize attention heatmap (conceptual)
    ax2.set_xlim(0, 512)
    ax2.set_ylim(512, 0)
    ax2.set_aspect('equal')
    
    # Create attention heatmap
    from matplotlib.patches import Circle
    
    # High attention areas (road markings)
    attention_points = [
        (206, 100, 50), (306, 100, 50),  # Vertical lines
        (75, 256, 50), (437, 256, 50),   # Horizontal lines
        (256, 100, 40), (256, 400, 40),  # Center line
    ]
    
    for x, y, radius in attention_points:
        circle = Circle((x, y), radius, color='red', alpha=0.3)
        ax2.add_patch(circle)
    
    # Draw the detected lines with arrows showing direction
    ax2.annotate('', xy=(206, 200), xytext=(206, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax2.annotate('', xy=(306, 200), xytext=(306, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax2.annotate('', xy=(150, 256), xytext=(0, 256),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    
    ax2.set_title("VLM Processing: Attention & Detection", fontsize=14)
    ax2.set_xlabel("X (pixels)")
    ax2.set_ylabel("Y (pixels)")
    ax2.text(256, 450, "Red: High attention areas\nBlue: Detected polylines", 
             ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    """Run the complete VLM demonstration"""
    
    # 1. Explain the pipeline
    explain_vlm_pipeline()
    
    # 2. Show image preprocessing
    print("\n" + "="*80)
    print("VISUALIZATION: Image Preprocessing")
    print("="*80)
    
    demo_image = "./internvl_chat/OpensatMap-demo/roundabout5_-1_-1_sat.png"
    if os.path.exists(demo_image):
        # For large image, use fewer patches for visualization
        fig1 = visualize_image_preprocessing(demo_image, max_num=6)
        plt.savefig('vlm_preprocessing.png', dpi=150, bbox_inches='tight')
        print("Preprocessing visualization saved to: vlm_preprocessing.png")
        plt.show()
    
    # 3. Demonstrate inference
    demonstrate_vlm_inference()
    
    # 4. Show attention concept
    print("\n" + "="*80)
    print("VISUALIZATION: VLM Attention Concept")
    print("="*80)
    fig2 = visualize_attention_concept()
    plt.savefig('vlm_attention_concept.png', dpi=150, bbox_inches='tight')
    print("Attention concept saved to: vlm_attention_concept.png")
    plt.show()
    
    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("1. InternVL is a Vision-Language Model that can understand images and text together")
    print("2. For road marking detection, it processes aerial images and outputs coordinate sequences")
    print("3. The model learns to associate visual patterns (white/yellow lines) with structured text output")
    print("4. Fine-tuning teaches the model the specific output format: <line> <x> <y> ... </line>")
    print("5. The VLM can handle complex scenes with multiple road markings and intersections")

if __name__ == "__main__":
    import os
    main()