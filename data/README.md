# Data

This directory should contain the PlantVillage dataset for training and evaluation.

## Required Data

### PlantVillage Dataset
- **Images**: 54,305 images across 38 classes
- **Source**: [Kaggle - PlantVillage Dataset](https://www.kaggle.com/datasets/divyasharma20/plantv)

### PlantVillageVQA Dataset
- **QA Pairs**: 193,609 question-answer pairs
- **Test Split**: 38,632 samples (in `splits/test_qa.json`)

## Download Instructions

### Option 1: Download from Kaggle (Recommended)

```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d divyasharma20/plantv
unzip plantv.zip -d data/
```

### Option 2: Direct Download

1. Visit [kaggle.com/datasets/divyasharma20/plantv](https://www.kaggle.com/datasets/divyasharma20/plantv)
2. Click **Download** button
3. Extract to `data/PlantVillage/`

## Expected Directory Structure

```
data/
├── PlantVillage/
│   ├── train/
│   │   ├── Apple___Apple_scab/
│   │   ├── Apple___Black_rot/
│   │   └── ... (38 classes)
│   └── val/
│       ├── Apple___Apple_scab/
│       ├── Apple___Black_rot/
│       └── ... (38 classes)
├── Images/                    # VQA images (55,448 images)
│   ├── image_0001.jpg
│   └── ...
└── splits/
    └── test_qa.json           # VQA test split
```

## Data Statistics

| Dataset | Train | Validation | Test |
|---------|-------|------------|------|
| PlantVillage | 43,456 | 10,850 | - |
| PlantVillageVQA | 135,526 | 19,351 | 38,632 |

## Test QA Format

The `test_qa.json` file contains:

```json
[
    {
        "image_id": "image_0001.jpg",
        "question": "What disease does this plant have?",
        "answer": "This plant has apple scab disease."
    },
    ...
]
```

## VQA Image Mapping

VQA images are mapped to PlantVillage classes. The mapping is stored in the VQA model checkpoint.
