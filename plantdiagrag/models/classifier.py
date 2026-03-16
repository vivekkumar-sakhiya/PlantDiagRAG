"""
ClassifierModel: Standalone classification model for plant disease identification.

Architecture: ViT-Base + BERT-Base with cross-modal attention
Note: This model uses T5-small dimensions (512) for compatibility with
the classifier checkpoint, unlike UnifiedPlantVLM which uses T5-base (768).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierModel(nn.Module):
    """
    Classifier model matching the training architecture.
    Used for standalone classification without VQA.
    """
    
    def __init__(self, vit_model, bert_model, num_classes=38):
        super().__init__()
        self.vit = vit_model
        self.bert = bert_model

        # Freeze base encoders
        for p in self.vit.parameters():
            p.requires_grad = False
        for p in self.bert.parameters():
            p.requires_grad = False

        # Unfreeze last 2 ViT layers
        for p in self.vit.encoder.layer[-2:].parameters():
            p.requires_grad = True

        # Projection layers (same as training)
        self.v_proj = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.t_proj = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self.norm = nn.LayerNorm(768)

        # T5 projection (512 dim for T5-small compatibility)
        self.to_t5 = nn.Linear(768, 512)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(768, 384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(384, num_classes)
        )

    def classify(self, pixel_values, input_ids, attention_mask):
        """Classification forward pass"""
        self.eval()
        with torch.no_grad():
            # Vision encoding
            v = self.vit(pixel_values).last_hidden_state
            v = torch.clamp(v, min=-10, max=10)
            v = self.v_proj(v)

            # Text encoding
            t = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
            t = torch.clamp(t, min=-10, max=10)
            t = self.t_proj(t)

            # Cross-modal fusion
            attn_out, _ = self.cross_attn(v, t, t)
            fused = self.norm(v + attn_out)

            # Classification
            logits = self.cls_head(fused[:, 0])
            probs = F.softmax(logits, dim=-1)

        return logits, probs

    @classmethod
    def from_pretrained(cls, vit_model, bert_model, checkpoint_path, device='cuda', num_classes=38):
        """Load a pretrained classifier model."""
        model = cls(vit_model, bert_model, num_classes=num_classes).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Filter out T5-related weights
        filtered_state = {k: v for k, v in state_dict.items()
                         if not k.startswith('t5.') and not k.startswith('bert.') and not k.startswith('vit.')}
        
        model.load_state_dict(filtered_state, strict=False)
        return model
