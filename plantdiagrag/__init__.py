"""
PlantDiagRAG: Unified Vision-Language Framework for Plant Disease Diagnosis

A unified framework combining:
- Disease Classification (99.10% accuracy)
- Visual Question Answering (VQA)
- RAG-based Treatment Recommendations

Paper: PlantDiagRAG: A Unified Vision-Language Framework for Plant Disease 
       Diagnosis and Treatment Recommendation
"""

from .pipeline import PlantDiagRAGPipeline
from .models.unified_vlm import UnifiedPlantVLM
from .models.classifier import ClassifierModel
from .rag.retriever import PlantDiseaseRAG

__version__ = "1.0.0"
__author__ = "Vivek Sakhiya"

__all__ = [
    'PlantDiagRAGPipeline',
    'UnifiedPlantVLM', 
    'ClassifierModel',
    'PlantDiseaseRAG'
]
