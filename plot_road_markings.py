import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import argparse

def read_json(file_path):
    """Read JSON annotation file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def plot_road_markings(image_path, json_path, output_path=None, show_plot=True):
    """
    Plot road markings from JSON annotations on the image.
    
    Args:
        image_path: Path to the satellite image
        json_path: Path to the JSON annotation file
        output_path: Optional path to save the annotated image
        show_plot: Whether to display the plot
    """
    # Read image
    img = mpimg.imread(image_path)
    
    # Read annotations
    json_data = read_json(json_path)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Display image
    ax.imshow(img)
    
    # Handle different JSON formats
    if isinstance(json_data, dict) and len(json_data) == 1:
        # Old format with image name as key
        img_name = list(json_data.keys())[0]
        lines = json_data[img_name]['lines']
    elif isinstance(json_data, list):
        # New format - direct list of polylines
        lines = []
        for line_coords in json_data:
            lines.append({'points': line_coords})
    else:
        raise ValueError("Unsupported JSON format")
    
    # Plot each line annotation
    for i, line in enumerate(lines):
        # Extract points
        if 'points' in line:
            points = np.array(line['points']).astype(np.int32)
        else:
            points = np.array(line).astype(np.int32)
        
        # Choose color based on line properties
        if isinstance(line, dict) and line.get('color') == 'White':
            color = 'white'
        elif isinstance(line, dict) and line.get('color') == 'Yellow':
            color = 'yellow'
        else:
            # Use colormap for visualization
            color = plt.cm.Set1(i % 9)
        
        # Determine line style based on line_type
        if line.get('line_type') == 'Dashed':
            linestyle = '--'
        else:
            linestyle = '-'
        
        # Plot the line
        ax.plot(points[:, 0], points[:, 1], 
                color=color, 
                linewidth=2, 
                linestyle=linestyle,
                alpha=0.8)
        
        # Add category label at the first point (optional)
        if i < 5:  # Only label first few lines to avoid clutter
            ax.text(points[0, 0], points[0, 1], 
                   f"{line.get('category', 'Line')}", 
                   fontsize=8, 
                   color='red',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
    
    ax.axis('off')
    ax.set_title(f"Road Markings - {img_name}")
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Annotated image saved to: {output_path}")
    
    if show_plot:
        plt.show()
    
    return fig

def plot_model_predictions(image_path, predictions, output_path=None, show_plot=True):
    """
    Plot road markings from model predictions.
    
    Args:
        image_path: Path to the satellite image
        predictions: List of polylines from model output
        output_path: Optional path to save the annotated image
        show_plot: Whether to display the plot
    """
    # Read image
    img = mpimg.imread(image_path)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Display image
    ax.imshow(img)
    
    # Plot each predicted line
    for i, polyline in enumerate(predictions):
        if len(polyline) > 1:
            points = np.array(polyline)
            # Use different colors for different lines
            color = plt.cm.rainbow(i / max(len(predictions), 1))
            ax.plot(points[:, 0], points[:, 1], 
                    color=color, 
                    linewidth=2, 
                    alpha=0.8,
                    marker='o',
                    markersize=3)
    
    ax.axis('off')
    ax.set_title(f"Model Predictions - {len(predictions)} lines detected")
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Predicted annotations saved to: {output_path}")
    
    if show_plot:
        plt.show()
    
    return fig

def parse_model_output(response):
    """
    Parse model output to extract polylines.
    Same function as in prompt_road_marking.py
    """
    polylines = []
    
    lines = response.strip().split('\n')
    
    for line in lines:
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
                    coords.append((x, y))
                except:
                    continue
            
            if coords:
                polylines.append(coords)
    
    return polylines

def main():
    parser = argparse.ArgumentParser(description='Plot road markings from annotations')
    parser.add_argument('--image', type=str, required=True, help='Path to satellite image')
    parser.add_argument('--json', type=str, help='Path to JSON annotation file')
    parser.add_argument('--model-output', type=str, help='Model output string with predictions')
    parser.add_argument('--output', type=str, help='Path to save annotated image')
    parser.add_argument('--no-show', action='store_true', help='Do not display the plot')
    
    args = parser.parse_args()
    
    if args.json:
        # Plot ground truth annotations
        plot_road_markings(args.image, args.json, args.output, not args.no_show)
    elif args.model_output:
        # Plot model predictions
        predictions = parse_model_output(args.model_output)
        plot_model_predictions(args.image, predictions, args.output, not args.no_show)
    else:
        print("Please provide either --json for ground truth or --model-output for predictions")

if __name__ == "__main__":
    main()