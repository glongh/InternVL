import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from internvl_chat.test import load_image

# Simple example to prompt a model with an image

# Configuration
model_path = "OpenGVLab/InternVL3-2B"  # Change to your fine-tuned model path
image_path = "./examples/image1.jpg"   # Change to your road marking image

# Road marking detection prompt (same as in your training data)
prompt = ("<image>\nFrom this aerial image of an urban street scene, identify and trace all visible road markings, "
          "including lane dividers, lane boundaries, bike lanes. For each marking, output a polyline or a sequence "
          "of (x, y) pixel coordinates representing its shape. Only include visible markings painted on the road surface.")

# Load model
print("Loading model...")
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map='auto'
).eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

# Load and prepare image
print("Loading image...")
pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()

# Generation settings
generation_config = dict(
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.7,
    top_p=0.95
)

# Get response
print("Generating response...")
response = model.chat(tokenizer, pixel_values, prompt, generation_config)

print(f"\nPrompt: {prompt}")
print(f"\nResponse: {response}")

# Parse the response to extract coordinates
def parse_coordinates(response):
    """Extract coordinate pairs from response"""
    lines = []
    for line in response.split('\n'):
        if '<line>' in line:
            # Extract numbers between <line> and </line>
            start = line.find('<line>') + 6
            end = line.find('</line>')
            if end > start:
                coords_str = line[start:end].strip()
                # Parse coordinate pairs
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

# Parse and display results
parsed_lines = parse_coordinates(response)
print(f"\nParsed {len(parsed_lines)} road markings:")
for i, line in enumerate(parsed_lines):
    print(f"  Line {i+1}: {line}")