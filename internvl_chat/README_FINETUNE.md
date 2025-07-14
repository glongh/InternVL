# InternVL3-2B Finetuning Guide

## Overview
This guide provides instructions for finetuning InternVL3-2B on custom datasets using either LoRA or full finetuning approaches.

## Prerequisites
- Conda environment `internvl` with flash-attn==2.3.6 installed
- CUDA 11.7+ and compatible GPU(s)
- Sufficient GPU memory (see requirements below)

## Scripts Created

### 1. LoRA Finetuning (`train_internvl3_2b_lora.sh`)
- **GPU Requirements**: 2-4 GPUs with ~20GB memory each
- **Key Features**:
  - Freezes vision encoder, LLM, and MLP
  - Uses LoRA rank 16 for LLM adaptation
  - Lower memory footprint
  - Faster training

### 2. Full Finetuning (`train_internvl3_2b_full.sh`)
- **GPU Requirements**: 8 GPUs with ~40GB memory each
- **Key Features**:
  - Trains LLM and MLP layers
  - Vision encoder frozen by default (can be unfrozen)
  - Better for significant task adaptation
  - Uses DeepSpeed Stage 2 for efficiency

## Dataset Format

### Configuration JSON (`dummy_dataset.json`)
```json
{
  "dataset_name": {
    "root": "/path/to/images/",
    "annotation": "/path/to/annotations.jsonl",
    "data_augment": false,
    "max_dynamic_patch": 12,
    "repeat_time": 1,
    "length": number_of_samples
  }
}
```

### Annotation JSONL Format
Each line should contain:
```json
{
  "id": unique_id,
  "image": "image_filename.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nYour question here"},
    {"from": "gpt", "value": "Model response here"}
  ]
}
```

## Usage

### LoRA Finetuning (Recommended for limited resources)
```bash
cd /home/paperspace/Developer/InternVL/internvl_chat
chmod +x train_internvl3_2b_lora.sh

# Run with default settings (2 GPUs)
./train_internvl3_2b_lora.sh

# Or customize GPU count and batch size
GPUS=4 PER_DEVICE_BATCH_SIZE=2 ./train_internvl3_2b_lora.sh
```

### Full Finetuning (For maximum performance)
```bash
cd /home/paperspace/Developer/InternVL/internvl_chat
chmod +x train_internvl3_2b_full.sh

# Run with default settings (8 GPUs)
./train_internvl3_2b_full.sh

# Or customize settings
GPUS=4 BATCH_SIZE=64 PER_DEVICE_BATCH_SIZE=4 ./train_internvl3_2b_full.sh
```

## Key Parameters

- `max_dynamic_patch`: 12 (splits high-res images into patches)
- `max_seq_length`: 8192 for LoRA, 16384 for full
- `learning_rate`: 4e-5 for LoRA, 2e-5 for full
- `freeze_backbone`: True (vision encoder frozen by default)

## Memory Optimization Tips

1. **Reduce batch size**: Lower `PER_DEVICE_BATCH_SIZE`
2. **Enable gradient checkpointing**: Already enabled in scripts
3. **Use DeepSpeed**: Stage 1 for LoRA, Stage 2/3 for full
4. **Reduce sequence length**: Lower `max_seq_length` if needed
5. **Use bf16**: Already enabled for efficiency

## Output
- Models saved to `work_dirs/internvl3_2b_{lora,full}_finetune/`
- Training logs saved to the same directory
- TensorBoard logs available for monitoring

## Troubleshooting

1. **OOM Errors**: Reduce batch size or use more GPUs
2. **CUDA Errors**: Ensure CUDA toolkit matches PyTorch version
3. **Dataset Errors**: Verify absolute paths in JSON config
4. **Import Errors**: Activate conda environment first

## Next Steps
1. Prepare your custom dataset in the required format
2. Update the dataset configuration JSON
3. Run the appropriate training script
4. Monitor training with TensorBoard
5. Evaluate the finetuned model on your tasks