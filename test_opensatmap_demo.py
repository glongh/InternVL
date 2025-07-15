#!/usr/bin/env python3
"""
Quick test script for OpenSatMap demo files
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Check if demo files exist
demo_dir = "./internvl_chat/OpensatMap-demo/"
image_file = demo_dir + "roundabout5_-1_-1_sat.png"
json_file = demo_dir + "RA.json"

print("Checking demo files...")
print(f"Image exists: {os.path.exists(image_file)}")
print(f"JSON exists: {os.path.exists(json_file)}")

if os.path.exists(image_file) and os.path.exists(json_file):
    # Load and display basic info
    img = mpimg.imread(image_file)
    print(f"\nImage shape: {img.shape}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    img_name = list(data.keys())[0]
    print(f"Image name in JSON: {img_name}")
    print(f"Number of line annotations: {len(data[img_name]['lines'])}")
    
    # Quick visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    
    # Plot first 10 lines
    for i, line in enumerate(data[img_name]['lines'][:10]):
        points = np.array(line['points'])
        color = 'yellow' if line.get('color') == 'Yellow' else 'white'
        plt.plot(points[:, 0], points[:, 1], color=color, linewidth=2, alpha=0.8)
    
    plt.title(f"First 10 road markings from {img_name}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('opensatmap_demo_test.png', dpi=150, bbox_inches='tight')
    print("\nTest visualization saved to: opensatmap_demo_test.png")
    plt.show()
else:
    print("\nDemo files not found. Please check the paths.")