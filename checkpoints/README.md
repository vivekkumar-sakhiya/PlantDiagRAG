# Model Checkpoints

This directory should contain the pre-trained model checkpoints. Due to their large size, they are not included in the Git repository.

## Required Checkpoints

| File | Size | Description |
|------|------|-------------|
| `best_vqa_model.pt` | ~1.76 GB | VQA model checkpoint |
| `best_classifier_v2.pt` | ~1.06 GB | Classification model checkpoint |

## Download Instructions

### Option 1: HuggingFace Hub (Recommended)

```bash
pip install huggingface_hub

# Download VQA model
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='vivek-sakhiya/PlantDiagRAG',
    filename='best_vqa_model.pt',
    local_dir='checkpoints'
)
"

# Download classifier model
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='vivek-sakhiya/PlantDiagRAG',
    filename='best_classifier_v2.pt',
    local_dir='checkpoints'
)
"
```

### Option 2: Direct Download

Download from HuggingFace Hub: https://huggingface.co/vivek-sakhiya/PlantDiagRAG

## Checkpoint Details

### VQA Model (`best_vqa_model.pt`)
- **Architecture**: UnifiedPlantVLM (ViT-Base + BERT-Base + Flan-T5-Base)
- **LoRA Config**: r=16, alpha=32, target_modules=["q", "v"]
- **Training**: 10 epochs on PlantVillageVQA
- **Val Loss**: 0.1236

### Classifier Model (`best_classifier_v2.pt`)
- **Architecture**: ClassifierModel (ViT-Base + BERT-Base)
- **Note**: Uses T5-small dimensions (512) for fusion layer
- **Training**: 8 epochs on PlantVillage
- **Val Accuracy**: 99.10%

## Checkpoint Contents

Both checkpoints contain:
```python
{
    'epoch': int,
    'model_state_dict': dict,
    'optimizer_state_dict': dict,  # (VQA only)
    'val_loss': float,             # (VQA only)
    'val_acc': float,              # (Classifier only)
    'label_mapping': dict,
    'idx_to_class': dict,
    'num_classes': 38
}
```

## Verification

After downloading, verify the checkpoints:

```python
import torch

# Check VQA model
vqa = torch.load('checkpoints/best_vqa_model.pt', map_location='cpu')
print(f"VQA Epoch: {vqa['epoch']}, Val Loss: {vqa['val_loss']:.4f}")

# Check classifier
cls = torch.load('checkpoints/best_classifier_v2.pt', map_location='cpu')
print(f"Classifier Epoch: {cls['epoch']}, Val Acc: {cls['val_acc']:.4f}")
```
