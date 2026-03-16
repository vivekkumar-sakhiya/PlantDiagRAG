# Data

This directory should contain the PlantVillage dataset for training and evaluation.

## Required Data

### PlantVillage Dataset
- **Images**: 54,306 images across 38 classes
- **Source**: [GitHub - spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset)

### PlantVillageVQA Dataset
- **QA Pairs**: 193,609 question-answer pairs
- **Test Split**: 38,632 samples (in `splits/test_qa.json`)

## Download Instructions

### Option 1: Download from Original Source

```bash
# Clone PlantVillage dataset
git clone https://github.com/spMohanty/PlantVillage-Dataset.git

# Copy the color images
cp -r PlantVillage-Dataset/raw/color data/PlantVillage
```

### Option 2: Download from Kaggle

```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d emmarex/plantdisease
unzip plantdisease.zip -d data/
```

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
