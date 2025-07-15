#!/usr/bin/env python3
"""
Demo script to visualize OpenSatMap road markings
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Paths to demo files
DEMO_DIR = "./internvl_chat/examples/satmap/"
IMAGE_PATH = DEMO_DIR + "patch_0044_896_1152.png"
JSON_PATH = DEMO_DIR + "patch_0044_896_1152.json"

# Read the image
print(f"Loading image from: {IMAGE_PATH}")
img = mpimg.imread(IMAGE_PATH)
print(f"Image shape: {img.shape}")

# Read the annotations
print(f"Loading annotations from: {JSON_PATH}")
with open(JSON_PATH, 'r') as f:
    json_data = json.load(f)

# Handle different JSON formats
if isinstance(json_data, dict) and len(json_data) == 1:
    # Old format with image name as key
    img_name = list(json_data.keys())[0]
    img_width = json_data[img_name]['image_width']
    img_height = json_data[img_name]['image_height']
    lines = json_data[img_name]['lines']
    print(f"Image: {img_name}")
    print(f"Dimensions: {img_width}x{img_height}")
elif isinstance(json_data, list):
    # New format - direct list of polylines
    lines = []
    for line_coords in json_data:
        lines.append({'points': line_coords})
    img_width, img_height = img.shape[1], img.shape[0]
    print(f"Image dimensions: {img_width}x{img_height}")
else:
    raise ValueError("Unsupported JSON format")

print(f"Number of line annotations: {len(lines)}")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Left: Original image
ax1.imshow(img)
ax1.set_title("Original Satellite Image")
ax1.axis('off')

# Right: Image with annotations
ax2.imshow(img)
ax2.set_title(f"Annotated Image - {len(lines)} road markings")
ax2.axis('off')

# Define color mapping for different line types
color_map = {
    'White': 'white',
    'Yellow': 'yellow',
    'Blue': 'blue',
    'Green': 'green'
}

line_style_map = {
    'Solid': '-',
    'Dashed': '--',
    'Dotted': ':'
}

# Plot each line
for i, line in enumerate(lines):
    # Extract points
    if 'points' in line:
        points = np.array(line['points']).astype(np.int32)
    else:
        points = np.array(line).astype(np.int32)
    
    # Get line properties
    if isinstance(line, dict):
        color_name = line.get('color', 'White')
        line_type = line.get('line_type', 'Solid')
        category = line.get('category', 'Lane line')
    else:
        color_name = 'White'
        line_type = 'Solid'
        category = 'Road marking'
    
    # Choose color
    if color_name in color_map:
        color = color_map[color_name]
    else:
        # Use random color
        np.random.seed(i)
        color = (np.random.rand(), np.random.rand(), np.random.rand())
    
    # Choose line style
    linestyle = line_style_map.get(line_type, '-')
    
    # Plot the line
    ax2.plot(points[:, 0], points[:, 1], 
             color=color, 
             linewidth=2, 
             linestyle=linestyle,
             alpha=0.8)
    
    # Optionally add start point marker
    ax2.plot(points[0, 0], points[0, 1], 'ro', markersize=3)

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='white', lw=2, label='White lanes'),
    Line2D([0], [0], color='yellow', lw=2, label='Yellow lanes'),
    Line2D([0], [0], color='red', lw=2, linestyle='--', label='Dashed lines'),
    Line2D([0], [0], marker='o', color='red', lw=0, markersize=5, label='Start points')
]
ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('road_markings_visualization.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved to: road_markings_visualization.png")
plt.show()

# Print some statistics
print("\nAnnotation Statistics:")
print("-" * 40)

# Count line types
line_types = {}
colors = {}
categories = {}

for line in lines:
    # Line type
    lt = line.get('line_type', 'Unknown')
    line_types[lt] = line_types.get(lt, 0) + 1
    
    # Color
    c = line.get('color', 'Unknown')
    colors[c] = colors.get(c, 0) + 1
    
    # Category
    cat = line.get('category', 'Unknown')
    categories[cat] = categories.get(cat, 0) + 1

print("Line Types:")
for lt, count in line_types.items():
    print(f"  {lt}: {count}")

print("\nColors:")
for c, count in colors.items():
    print(f"  {c}: {count}")

print("\nCategories:")
for cat, count in categories.items():
    print(f"  {cat}: {count}")

# Example of how model output would be parsed
print("\n" + "="*60)
print("Example: Parsing model output format")
print("="*60)

# Simulate model output
example_model_output = """<line> <1365> <1920> <1346> <1945> <1319> <1970> <1265> <2013> </line>
<line> <2048> <1536> <2100> <1550> <2150> <1565> <2200> <1580> </line>"""

def parse_coordinates(response):
    """Extract coordinate pairs from model response"""
    lines = []
    for line in response.strip().split('\n'):
        if '<line>' in line:
            start = line.find('<line>') + 6
            end = line.find('</line>')
            if end > start:
                coords_str = line[start:end].strip()
                coords = []
                parts = coords_str.split()
                for i in range(0, len(parts)-1, 2):
                    try:
                        x = int(parts[i].replace('<', '').replace('>', ''))
                        y = int(parts[i+1].replace('<', '').replace('>', ''))
                        coords.append((x, y))
                    except:
                        pass
                if coords:
                    lines.append(coords)
    return lines

parsed_lines = parse_coordinates(example_model_output)
print(f"Parsed {len(parsed_lines)} lines from model output:")
for i, line in enumerate(parsed_lines):
    print(f"  Line {i+1}: {len(line)} points - {line[:2]}...")