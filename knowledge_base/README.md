# Knowledge Base

This directory contains the agricultural knowledge base used for RAG-based treatment recommendations.

## Required File

| File | Description |
|------|-------------|
| `all_kb_documents.json` | 54 documents from agricultural sources |

## Knowledge Base Sources

| Source | Documents | Percentage |
|--------|-----------|------------|
| ICAR (Indian Council of Agricultural Research) | 18 | 33.3% |
| UC IPM (UC Integrated Pest Management) | 15 | 27.8% |
| PNW Handbook (Pacific Northwest Plant Disease) | 12 | 22.2% |
| AGROVOC (FAO Agricultural Vocabulary) | 9 | 16.7% |

**Total**: 54 documents covering 78.9% of the 38 disease classes.

## Document Format

Each document in `all_kb_documents.json` has the following structure:

```json
{
    "title": "Apple Disease Management Guidelines",
    "content": "APPLE SCAB (Venturia inaequalis):\n- Most important disease...",
    "source": "ICAR (ICAR-CITH)",
    "disease_coverage": ["Apple___Apple_scab", "Apple___Black_rot", ...],
    "url": "https://..."
}
```

## Disease Coverage

### Fully Covered (Specific Treatment Info)
- All Apple diseases
- All Tomato diseases
- All Grape diseases
- Potato Early/Late Blight
- Corn diseases
- Pepper Bacterial Spot

### Partially Covered (General Info)
- Peach Bacterial Spot
- Strawberry Leaf Scorch
- Squash Powdery Mildew

### Healthy Classes
Healthy plant classes return general maintenance recommendations.

## Building Your Own Knowledge Base

To add custom documents:

```python
import json

# Load existing KB
with open('all_kb_documents.json', 'r') as f:
    kb = json.load(f)

# Add new document
kb.append({
    "title": "Your Document Title",
    "content": "Your document content...",
    "source": "Your Source",
    "url": "https://your-source.com"
})

# Save
with open('all_kb_documents.json', 'w') as f:
    json.dump(kb, f, indent=2)
```

## FAISS Index

The RAG system automatically builds a FAISS index from the knowledge base on first load. The index uses:
- **Embedding Model**: `all-MiniLM-L6-v2`
- **Index Type**: `IndexFlatIP` (Inner Product / Cosine Similarity)
- **Dimension**: 384

## Download
- HuggingFace: https://huggingface.co/vivek-sakhiya/PlantDiagRAG
