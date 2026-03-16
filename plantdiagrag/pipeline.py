"""
PlantDiagRAG Pipeline: Complete plant disease diagnosis system.

Combines:
1. Classification - Disease identification
2. VQA - Question answering about plant diseases
3. RAG - Treatment recommendations from knowledge base
"""

import os
import json
import torch
from PIL import Image
from transformers import ViTModel, ViTImageProcessor, BertModel, BertTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import LoraConfig, get_peft_model, TaskType

from .models.unified_vlm import UnifiedPlantVLM
from .models.classifier import ClassifierModel
from .rag.retriever import PlantDiseaseRAG


class PlantDiagRAGPipeline:
    """
    Complete plant disease diagnosis pipeline.
    
    Example usage:
        pipeline = PlantDiagRAGPipeline.from_pretrained(
            vqa_checkpoint="path/to/best_vqa_model.pt",
            classifier_checkpoint="path/to/best_classifier_v2.pt",
            knowledge_base="path/to/all_kb_documents.json",
            label_mapping="path/to/label_mapping.json"
        )
        
        result = pipeline.diagnose("path/to/plant_image.jpg", question="What disease is this?")
    """
    
    def __init__(
        self,
        vqa_model,
        classifier_model,
        rag_system,
        vit_processor,
        bert_tokenizer,
        t5_tokenizer,
        label_mapping,
        idx_to_class,
        device='cuda'
    ):
        self.vqa_model = vqa_model
        self.classifier_model = classifier_model
        self.rag = rag_system
        self.vit_proc = vit_processor
        self.bert_tok = bert_tokenizer
        self.t5_tok = t5_tokenizer
        self.label_mapping = label_mapping
        self.idx_to_class = idx_to_class
        self.device = device
        
    @classmethod
    def from_pretrained(
        cls,
        vqa_checkpoint,
        classifier_checkpoint,
        knowledge_base,
        label_mapping_path,
        device='cuda'
    ):
        """
        Load a pretrained PlantDiagRAG pipeline.
        
        Args:
            vqa_checkpoint: Path to VQA model checkpoint
            classifier_checkpoint: Path to classifier checkpoint
            knowledge_base: Path to knowledge base JSON
            label_mapping_path: Path to label mapping JSON
            device: Device to use ('cuda' or 'cpu')
        """
        print("🔄 Loading PlantDiagRAG pipeline...")
        
        # Load label mapping
        with open(label_mapping_path, 'r') as f:
            mapping_data = json.load(f)
        label_mapping = mapping_data['label_mapping']
        idx_to_class = {int(k): v for k, v in mapping_data['idx_to_class'].items()}
        num_classes = mapping_data['num_classes']
        
        # Load base models
        print("  Loading ViT...")
        vit_proc = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        vit = ViTModel.from_pretrained('google/vit-base-patch16-224').to(device)
        
        print("  Loading BERT...")
        bert_tok = BertTokenizer.from_pretrained('bert-base-uncased')
        bert = BertModel.from_pretrained('bert-base-uncased').to(device)
        
        print("  Loading T5 with LoRA...")
        t5_tok = T5Tokenizer.from_pretrained('google/flan-t5-base')
        t5 = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base').to(device)
        
        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q", "v"]
        )
        t5 = get_peft_model(t5, lora_config)
        
        # Create and load VQA model
        print("  Loading VQA model...")
        vqa_model = UnifiedPlantVLM(vit, bert, t5, num_classes=num_classes).to(device)
        vqa_ckpt = torch.load(vqa_checkpoint, map_location=device)
        vqa_state = vqa_ckpt.get('model_state_dict', vqa_ckpt)
        vqa_model.load_state_dict(vqa_state, strict=False)
        vqa_model.eval()
        
        # Create and load classifier model
        print("  Loading classifier model...")
        classifier_model = ClassifierModel(vit, bert, num_classes=num_classes).to(device)
        cls_ckpt = torch.load(classifier_checkpoint, map_location=device)
        cls_state = cls_ckpt.get('model_state_dict', cls_ckpt)
        filtered_state = {k: v for k, v in cls_state.items()
                         if not k.startswith('t5.') and not k.startswith('bert.') and not k.startswith('vit.')}
        classifier_model.load_state_dict(filtered_state, strict=False)
        classifier_model.eval()
        
        # Load RAG system
        print("  Loading RAG system...")
        rag_system = PlantDiseaseRAG(knowledge_base)
        
        print("✓ Pipeline loaded successfully!")
        
        return cls(
            vqa_model=vqa_model,
            classifier_model=classifier_model,
            rag_system=rag_system,
            vit_processor=vit_proc,
            bert_tokenizer=bert_tok,
            t5_tokenizer=t5_tok,
            label_mapping=label_mapping,
            idx_to_class=idx_to_class,
            device=device
        )
    
    def diagnose(self, image_path_or_pil, question=None):
        """
        Complete plant disease diagnosis.
        
        Args:
            image_path_or_pil: Path to image or PIL Image
            question: Optional question for VQA
            
        Returns:
            Dictionary with diagnosis results:
            - classification: Disease prediction with confidence
            - vqa: Answer to the question (if provided)
            - treatment: Treatment recommendations from RAG
        """
        # Load image
        if isinstance(image_path_or_pil, str):
            img = Image.open(image_path_or_pil).convert('RGB')
        else:
            img = image_path_or_pil.convert('RGB')

        # Process image
        pixel_values = self.vit_proc(img, return_tensors='pt')['pixel_values'].to(self.device)

        results = {
            'classification': {},
            'vqa': {},
            'treatment': {}
        }

        # ─────────────────────────────────────────────────────────────
        # STEP 1: CLASSIFICATION
        # ─────────────────────────────────────────────────────────────
        cls_prompt = "Classify the plant disease in this image."
        cls_enc = self.bert_tok(cls_prompt, max_length=64, padding='max_length',
                                truncation=True, return_tensors='pt')

        logits, probs = self.classifier_model.classify(
            pixel_values,
            cls_enc['input_ids'].to(self.device),
            cls_enc['attention_mask'].to(self.device)
        )

        pred_idx = probs.argmax().item()
        pred_class = self.idx_to_class[pred_idx]
        confidence = probs.max().item()

        # Get top 3 predictions
        top3_probs, top3_idx = probs[0].topk(3)
        top3 = [(self.idx_to_class[idx.item()], prob.item()) 
                for prob, idx in zip(top3_probs, top3_idx)]

        results['classification'] = {
            'predicted_class': pred_class,
            'confidence': confidence,
            'top3': top3,
            'is_healthy': 'healthy' in pred_class.lower()
        }

        # ─────────────────────────────────────────────────────────────
        # STEP 2: VQA (if question provided)
        # ─────────────────────────────────────────────────────────────
        if question:
            q_enc = self.bert_tok(f"Question: {question}", max_length=128, 
                                  padding='max_length', truncation=True, return_tensors='pt')

            output_ids = self.vqa_model.generate(
                pixel_values,
                q_enc['input_ids'].to(self.device),
                q_enc['attention_mask'].to(self.device),
                max_length=128
            )

            answer = self.t5_tok.decode(output_ids[0], skip_special_tokens=True)
            
            results['vqa'] = {
                'question': question,
                'answer': answer
            }
        else:
            # Default question
            default_q = "What is wrong with this plant and how can I treat it?"
            q_enc = self.bert_tok(f"Question: {default_q}", max_length=128,
                                  padding='max_length', truncation=True, return_tensors='pt')

            output_ids = self.vqa_model.generate(
                pixel_values,
                q_enc['input_ids'].to(self.device),
                q_enc['attention_mask'].to(self.device),
                max_length=128
            )

            answer = self.t5_tok.decode(output_ids[0], skip_special_tokens=True)
            
            results['vqa'] = {
                'question': default_q,
                'answer': answer
            }

        # ─────────────────────────────────────────────────────────────
        # STEP 3: RAG TREATMENT RETRIEVAL
        # ─────────────────────────────────────────────────────────────
        treatment_query = f"Treatment for {pred_class.replace('___', ' ').replace('_', ' ')}"
        treatment_result = self.rag.get_treatment_info(treatment_query, disease_class=pred_class)

        results['treatment'] = {
            'summary': treatment_result['treatment_summary'],
            'sources': treatment_result['sources'],
            'documents': treatment_result['documents']
        }

        return results
    
    def classify(self, image_path_or_pil):
        """
        Classify plant disease only.
        
        Args:
            image_path_or_pil: Path to image or PIL Image
            
        Returns:
            Dictionary with classification results
        """
        result = self.diagnose(image_path_or_pil)
        return result['classification']
    
    def answer_question(self, image_path_or_pil, question):
        """
        Answer a question about the plant image.
        
        Args:
            image_path_or_pil: Path to image or PIL Image
            question: Question to answer
            
        Returns:
            Generated answer string
        """
        result = self.diagnose(image_path_or_pil, question=question)
        return result['vqa']['answer']
    
    def get_treatment(self, disease_class):
        """
        Get treatment recommendations for a disease.
        
        Args:
            disease_class: Disease class name (e.g., "Tomato___Late_blight")
            
        Returns:
            Treatment information dictionary
        """
        query = f"Treatment for {disease_class.replace('___', ' ').replace('_', ' ')}"
        return self.rag.get_treatment_info(query, disease_class=disease_class)
