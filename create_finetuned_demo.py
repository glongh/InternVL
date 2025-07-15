#!/usr/bin/env python3
"""
Create a demo showing what a fine-tuned model would output
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

def convert_polylines_to_model_format(polylines):
    """Convert ground truth polylines to the model output format"""
    output_lines = []
    
    for polyline in polylines:
        if len(polyline) > 1:
            # Convert to integers and create the line format
            coords = []
            for point in polyline:
                x, y = int(point[0]), int(point[1])
                coords.extend([f"<{x}>", f"<{y}>"])
            
            line_str = f"<line> {' '.join(coords)} </line>"
            output_lines.append(line_str)
    
    return "\n".join(output_lines)

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
            
            if len(coords) > 1:
                polylines.append(np.array(coords))
    
    return polylines

def create_finetuned_demo_visualization(image_path, gt_polylines, output_path):
    """Create demo showing original, ground truth, and simulated fine-tuned output"""
    
    # Load image
    img = mpimg.imread(image_path)
    
    # Create simulated fine-tuned output (using ground truth as example)
    simulated_output = convert_polylines_to_model_format(gt_polylines)
    simulated_polylines = parse_model_predictions(simulated_output)
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
    
    # Original image
    ax1.imshow(img)
    ax1.set_title("Original Satellite Image", fontsize=14)
    ax1.axis('off')
    
    # Ground truth
    ax2.imshow(img, alpha=0.7)
    ax2.set_title(f"Ground Truth Annotations\n({len(gt_polylines)} road markings)", fontsize=14)
    ax2.axis('off')
    
    for i, line in enumerate(gt_polylines):
        if len(line) > 1:
            ax2.plot(line[:, 0], line[:, 1], 
                     color='lime', 
                     linewidth=2, 
                     alpha=0.8,
                     marker='o',
                     markersize=1)
    
    # Simulated fine-tuned model output
    ax3.imshow(img, alpha=0.7)
    ax3.set_title(f"Fine-tuned Model Output\n({len(simulated_polylines)} detected markings)", fontsize=14)
    ax3.axis('off')
    
    for i, line in enumerate(simulated_polylines):
        if len(line) > 1:
            color = plt.cm.rainbow(i / max(len(simulated_polylines), 1))
            ax3.plot(line[:, 0], line[:, 1], 
                     color=color, 
                     linewidth=2, 
                     alpha=0.8,
                     marker='o',
                     markersize=1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return simulated_output

def create_comparison_all_images():
    """Create comparison for all images"""
    
    # Directory paths
    satmap_dir = "/home/paperspace/Developer/InternVL/internvl_chat/examples/satmap"
    output_dir = "/home/paperspace/Developer/InternVL/finetuned_demo_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(satmap_dir) if f.endswith('.png')]
    image_files.sort()
    
    print(f"Creating fine-tuned model demos for {len(image_files)} images")
    
    # Process each image
    for i, img_file in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] Processing: {img_file}")
        
        # File paths
        image_path = os.path.join(satmap_dir, img_file)
        json_path = os.path.join(satmap_dir, img_file.replace('.png', '.json'))
        
        # Load ground truth
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        gt_polylines = []
        for line in data:
            if len(line) > 1:
                gt_polylines.append(np.array(line))
        
        print(f"  Ground truth: {len(gt_polylines)} polylines")
        
        # Create demo visualization
        base_name = img_file.replace('.png', '')
        demo_path = os.path.join(output_dir, f"{base_name}_finetuned_demo.png")
        
        simulated_output = create_finetuned_demo_visualization(image_path, gt_polylines, demo_path)
        
        # Save simulated model output
        output_path = os.path.join(output_dir, f"{base_name}_simulated_output.txt")
        with open(output_path, 'w') as f:
            f.write("SIMULATED FINE-TUNED MODEL OUTPUT:\n")
            f.write("="*50 + "\n\n")
            f.write(simulated_output)
        
        print(f"  ✅ Demo created: {demo_path}")
    
    print(f"\n✅ All demos created! Results saved to: {output_dir}")

def show_format_comparison():
    """Show the difference between base model and fine-tuned model output"""
    
    print("="*80)
    print("COMPARISON: BASE MODEL vs FINE-TUNED MODEL OUTPUT")
    print("="*80)
    
    print("\n📝 BASE MODEL OUTPUT (Current):")
    print("-"*40)
    print("""To identify and trace the road markings in the aerial image, we need to focus on the visible lane dividers and boundaries. Here are the details:

1. **Lane Dividers:**
   - **Lane Divider (Yellow and White):** This is a dashed yellow line running along the center of the road, separating the lanes.
     - **Coordinates:** (Approximate coordinates for the dashed yellow line: (100, 150) to (900, 150))

2. **Lane Boundaries:**
   - **Lane Boundaries (White):** These are solid white lines marking the edges of the lanes.
     - **Coordinates:** (Approximate coordinates for the white lane boundaries: (100, 150) to (100, 200) and (900, 150) to (900, 200))""")
    
    print("\n🎯 FINE-TUNED MODEL OUTPUT (Target):")
    print("-"*40)
    print("""<line> <512> <385> <316> <395> <215> <400> <31> <409> <0> <411> </line>
<line> <140> <492> <401> <477> <512> <470> </line>
<line> <116> <506> <131> <504> <190> <499> <231> <495> <291> <493> <408> <485> <512> <480> </line>
<line> <0> <503> <78> <496> <140> <492> </line>
<line> <512> <373> <433> <378> <218> <390> <0> <402> </line>""")
    
    print("\n💡 KEY DIFFERENCES:")
    print("-"*40)
    print("• Base model: Gives descriptive text with approximate coordinates")
    print("• Fine-tuned model: Outputs precise pixel coordinates in structured format")
    print("• Base model: Human-readable but not machine-parseable")
    print("• Fine-tuned model: Machine-parseable coordinate sequences")
    print("• Base model: Identifies general categories")
    print("• Fine-tuned model: Traces exact polylines for each marking")
    print()

if __name__ == "__main__":
    # Show format comparison
    show_format_comparison()
    
    # Create demo visualizations
    create_comparison_all_images()