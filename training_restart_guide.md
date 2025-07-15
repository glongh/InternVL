# Fine-tuning Training Restart Guide

## Current Status
- Training checkpoint at step 10,200 is corrupted
- Model weights (model.safetensors) cannot be loaded
- Need to restart training from scratch

## Steps to Restart Training

### 1. Prepare Data
```bash
# Make sure your training data is in the correct format
# Each line should be a JSON object with:
# - "image": path to image file
# - "conversations": [{"from": "human", "value": "prompt"}, {"from": "gpt", "value": "coordinates"}]
```

### 2. Check Training Configuration
```bash
# Review your training script/config
# Key parameters to verify:
# - learning_rate: 4e-05 (from previous run)
# - batch_size: 4 (per device)
# - gradient_accumulation_steps: 2
# - num_epochs: 1.0
```

### 3. Restart Training
```bash
# Run your training script again
# Consider:
# - Save checkpoints more frequently (every 1000 steps)
# - Use .pth format instead of safetensors
# - Monitor GPU memory usage
# - Check for any data loading errors
```

### 4. Test Intermediate Checkpoints
```bash
# Test checkpoints during training
python test_finetuned_checkpoint.py
```

### 5. Expected Output Format
The model should output:
```
<line> <x1> <y1> <x2> <y2> ... </line>
<line> <x1> <y1> <x2> <y2> ... </line>
```

Instead of descriptive text.
