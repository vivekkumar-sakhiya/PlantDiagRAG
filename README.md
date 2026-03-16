# 🌱 PlantDiagRAG

**A Unified Vision-Language Framework for Plant Disease Diagnosis and Treatment Recommendation**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PlantDiagRAG is a unified framework that combines **disease classification**, **visual question answering (VQA)**, and **RAG-based treatment recommendations** into a single system for comprehensive plant disease diagnosis.

## 🎯 Key Features

- **Disease Classification**: 99.10% accuracy across 38 plant disease classes
- **Visual Question Answering**: Answer questions about plant diseases using vision-language understanding
- **Treatment Recommendations**: Evidence-based treatment suggestions from agricultural knowledge bases (ICAR, UC IPM, PNW Handbook, AGROVOC)
- **Unified Architecture**: Single model for multiple tasks using ViT + BERT + Flan-T5 with LoRA

## 📊 Performance

| Task | Metric | Score |
|------|--------|-------|
| Classification | Accuracy | **99.10%** |
| Classification | Precision | 99.16% |
| Classification | F1-Score | 99.11% |
| VQA | BLEU | 21.63% |
| VQA | ROUGE-1 | 64.63% |
| VQA | ROUGE-L | 63.38% |
| VQA | METEOR | 44.05% |
| VQA | Token F1 | 67.82% |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      PlantDiagRAG                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐    │
│  │   ViT-Base  │   │  BERT-Base  │   │ Flan-T5 + LoRA  │    │
│  │   (Vision)  │   │   (Text)    │   │   (Generation)  │    │
│  └──────┬──────┘   └──────┬──────┘   └─────────┬───────┘    │
│         │                 │                    │            │
│         └────────┬────────┘                    │            │
│                  ▼                             │            │
│         ┌───────────────┐                      │            │
│         │ Cross-Modal   │──────────────────────┘            │
│         │   Attention   │                                   │
│         └───────┬───────┘                                   │
│                 │                                           │
│    ┌────────────┼────────────┐                              │
│    ▼            ▼            ▼                              │
│ ┌──────┐   ┌─────────┐   ┌─────────┐                        │
│ │Class │   │   VQA   │   │   RAG   │                        │
│ │ Head │   │ Output  │   │Retrieval│                        │
│ └──────┘   └─────────┘   └─────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/viveksakhiya/PlantDiagRAG.git
cd PlantDiagRAG

# Install dependencies
pip install -r requirements.txt

# Download model checkpoints (see Checkpoints section)
```

### Basic Usage

```python
from plantdiagrag import PlantDiagRAGPipeline

# Load the pipeline
pipeline = PlantDiagRAGPipeline.from_pretrained(
    vqa_checkpoint="checkpoints/best_vqa_model.pt",
    classifier_checkpoint="checkpoints/best_classifier_v2.pt",
    knowledge_base="knowledge_base/all_kb_documents.json",
    label_mapping_path="configs/label_mapping.json",
    device="cuda"
)

# Diagnose a plant image
result = pipeline.diagnose("path/to/plant_image.jpg")

# Access results
print(f"Disease: {result['classification']['predicted_class']}")
print(f"Confidence: {result['classification']['confidence']:.2%}")
print(f"Treatment: {result['treatment']['summary']}")
```

### Command Line Interface

```bash
# Single image diagnosis
python scripts/inference.py --image path/to/image.jpg

# With custom question
python scripts/inference.py --image path/to/image.jpg --question "What are the symptoms?"

# Batch processing
python scripts/inference.py --image_dir path/to/images/ --output results.json
```

### Gradio Web Demo

```bash
python scripts/demo_gradio.py --share
```

## 📁 Repository Structure

```
PlantDiagRAG/
├── plantdiagrag/              # Main package
│   ├── models/                # Model architectures
│   │   ├── unified_vlm.py     # UnifiedPlantVLM (VQA model)
│   │   └── classifier.py      # ClassifierModel
│   ├── rag/                   # RAG components
│   │   └── retriever.py       # PlantDiseaseRAG
│   ├── evaluation/            # Evaluation metrics
│   │   └── metrics.py         # BLEU, ROUGE, METEOR, etc.
│   └── pipeline.py            # Main pipeline
├── configs/                   # Configuration files
│   └── label_mapping.json     # Class name mappings
├── knowledge_base/            # RAG knowledge base
│   └── all_kb_documents.json  # 54 agricultural documents
├── checkpoints/               # Model checkpoints (download separately)
├── scripts/                   # CLI scripts
│   ├── inference.py           # Command line inference
│   └── demo_gradio.py         # Gradio web demo
├── notebooks/                 # Jupyter notebooks
├── requirements.txt
└── README.md
```

## 📦 Checkpoints

Model checkpoints are hosted on HuggingFace Hub due to their size:

| Model | Size | Description |
|-------|------|-------------|
| `best_vqa_model.pt` | ~1.76 GB | VQA model (ViT + BERT + T5-Base + LoRA) |
| `best_classifier_v2.pt` | ~1.06 GB | Classifier model (ViT + BERT + T5-Small) |

**Download from HuggingFace:**
```bash
# Using huggingface_hub
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('viveksakhiya/PlantDiagRAG', 'best_vqa_model.pt', local_dir='checkpoints')"
```

Or download directly from: [HuggingFace Hub](https://huggingface.co/viveksakhiya/PlantDiagRAG)

## 📚 Datasets

### PlantVillage
- **Images**: 54,306 images across 38 classes
- **Source**: [PlantVillage Dataset](https://www.kaggle.com/datasets/divyasharma20/plantv/data)

### PlantVillageVQA
- **QA Pairs**: 193,609 question-answer pairs
- **Source**: [PlantVillageVQA Dataset](https://huggingface.co/datasets/SyedNazmusSakib/PlantVillageVQA)

### Knowledge Base
- **Documents**: 54 agricultural documents
- **Sources**: ICAR (33.3%), UC IPM (27.8%), PNW Handbook (22.2%), AGROVOC (16.7%)
- **Coverage**: 78.9% of disease classes

## 🔬 Supported Disease Classes

<details>
<summary>Click to expand all 38 classes</summary>

1. Apple___Apple_scab
2. Apple___Black_rot
3. Apple___Cedar_apple_rust
4. Apple___healthy
5. Blueberry___healthy
6. Cherry_(including_sour)___Powdery_mildew
7. Cherry_(including_sour)___healthy
8. Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot
9. Corn_(maize)___Common_rust_
10. Corn_(maize)___Northern_Leaf_Blight
11. Corn_(maize)___healthy
12. Grape___Black_rot
13. Grape___Esca_(Black_Measles)
14. Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
15. Grape___healthy
16. Orange___Haunglongbing_(Citrus_greening)
17. Peach___Bacterial_spot
18. Peach___healthy
19. Pepper,_bell___Bacterial_spot
20. Pepper,_bell___healthy
21. Potato___Early_blight
22. Potato___Late_blight
23. Potato___healthy
24. Raspberry___healthy
25. Soybean___healthy
26. Squash___Powdery_mildew
27. Strawberry___Leaf_scorch
28. Strawberry___healthy
29. Tomato___Bacterial_spot
30. Tomato___Early_blight
31. Tomato___Late_blight
32. Tomato___Leaf_Mold
33. Tomato___Septoria_leaf_spot
34. Tomato___Spider_mites_Two-spotted_spider_mite
35. Tomato___Target_Spot
36. Tomato___Tomato_Yellow_Leaf_Curl_Virus
37. Tomato___Tomato_mosaic_virus
38. Tomato___healthy

</details>

## 📄 Citation

If you use PlantDiagRAG in your research, please cite:

```bibtex
@article{sakhiya2026plantdiagrag,
  title={PlantDiagRAG: A Unified Vision-Language Framework for Plant Disease Diagnosis and Treatment Recommendation},
  author={Vivekkumar Sakhiya and Dr. Abhinav Kumar},
  journal={},
  year={2026}
}
```


## 🙏 Acknowledgments

- [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset) for the plant disease images
- [ICAR](https://icar.org.in/) for comprehensive disease management guidelines
- [UC IPM](https://ipm.ucanr.edu/) for integrated pest management resources
- [PNW Handbook](https://pnwhandbooks.org/) for Pacific Northwest plant disease information
- [AGROVOC](https://www.fao.org/agrovoc/) for agricultural vocabulary and concepts

## 📧 Contact

- **Author**: Vivekkumar Sakhiya
- **Email**: [vivekskahiya369@gmail.com]
- **Supervisor**: Dr. Abhinav Kumar, MNNIT Allahabad

---

<p align="center">Made with ❤️ for sustainable agriculture</p>
