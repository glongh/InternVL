import argparse
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoConfig
import json

# Import required functions from test.py
from internvl_chat.test import load_image, split_model

def prompt_model_with_image(model_path, image_path, prompt=None, device='cuda', load_in_8bit=False):
    """
    Prompt a model checkpoint with an image.
    
    Args:
        model_path: Path to the model checkpoint
        image_path: Path to the image file
        prompt: Custom prompt (if None, uses default road marking prompt)
        device: Device to run on ('cuda' or 'cpu')
        load_in_8bit: Whether to load model in 8-bit quantization
    
    Returns:
        Model response string
    """
    # Default prompt for road marking detection
    if prompt is None:
        prompt = ("<image>\nFrom this aerial image of an urban street scene, identify and trace all visible road markings, "
                 "including lane dividers, lane boundaries, bike lanes. For each marking, output a polyline or a sequence "
                 "of (x, y) pixel coordinates representing its shape. Only include visible markings painted on the road surface.")
    
    # Setup device map for multi-GPU if available
    if torch.cuda.device_count() > 1:
        device_map = split_model(model_path)
    else:
        device_map = 'auto' if device == 'cuda' else None
    
    # Load model and tokenizer
    print(f"Loading model from: {model_path}")
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=load_in_8bit,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    
    # Load and process image
    print(f"Loading image from: {image_path}")
    pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16)
    if device == 'cuda':
        pixel_values = pixel_values.cuda()
    
    # Generation configuration
    generation_config = dict(
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )
    
    # Get model response
    print("Generating response...")
    response = model.chat(tokenizer, pixel_values, prompt, generation_config)
    
    return response

def parse_road_markings(response):
    """
    Parse the model response to extract road marking coordinates.
    
    Args:
        response: Model response string containing road markings
    
    Returns:
        List of polylines, where each polyline is a list of (x, y) tuples
    """
    polylines = []
    
    # Split response into lines
    lines = response.strip().split('\n')
    
    for line in lines:
        if '<line>' in line and '</line>' in line:
            # Extract coordinates between <line> tags
            start = line.find('<line>') + 6
            end = line.find('</line>')
            coords_str = line[start:end].strip()
            
            # Parse coordinates
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
    parser = argparse.ArgumentParser(description='Prompt InternVL model with an image')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image-path', type=str, required=True, help='Path to input image')
    parser.add_argument('--prompt', type=str, default=None, help='Custom prompt (default: road marking detection)')
    parser.add_argument('--output-json', type=str, help='Path to save output in JSON format')
    parser.add_argument('--load-in-8bit', action='store_true', help='Load model in 8-bit quantization')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to run on')
    
    args = parser.parse_args()
    
    # Get model response
    response = prompt_model_with_image(
        model_path=args.model_path,
        image_path=args.image_path,
        prompt=args.prompt,
        device=args.device,
        load_in_8bit=args.load_in_8bit
    )
    
    print(f"\nModel Response:\n{response}")
    
    # Parse road markings if using default prompt
    if args.prompt is None:
        polylines = parse_road_markings(response)
        print(f"\nParsed {len(polylines)} road markings:")
        for i, polyline in enumerate(polylines):
            print(f"  Marking {i+1}: {len(polyline)} points")
    
    # Save output if requested
    if args.output_json:
        output_data = {
            "image_path": args.image_path,
            "prompt": args.prompt if args.prompt else "default_road_marking_prompt",
            "response": response,
            "width": 512,  # Default from sample, can be made dynamic
            "height": 512
        }
        
        # Add parsed polylines if using default prompt
        if args.prompt is None:
            output_data["parsed_polylines"] = polylines
        
        with open(args.output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nOutput saved to: {args.output_json}")

if __name__ == "__main__":
    main()