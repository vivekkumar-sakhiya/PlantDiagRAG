"""
UnifiedPlantVLM: Unified Vision-Language Model for Plant Disease
- Classification: Disease identification
- VQA: Question answering about plant diseases

Architecture: ViT-Base + BERT-Base + Flan-T5-Base with LoRA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput


class UnifiedPlantVLM(nn.Module):
    """
    Unified Vision-Language Model for Plant Disease
    - Classification: Disease identification
    - VQA: Question answering about plant diseases

    Architecture matches Phase 2 training exactly.
    """

    def __init__(self, vit_model, bert_model, t5_model, num_classes=38):
        super().__init__()
        self.vit = vit_model
        self.bert = bert_model
        self.t5 = t5_model

        # Freeze base encoders
        for p in self.vit.parameters():
            p.requires_grad = False
        for p in self.bert.parameters():
            p.requires_grad = False

        # Unfreeze last 2 ViT layers
        for p in self.vit.encoder.layer[-2:].parameters():
            p.requires_grad = True

        # Projection layers
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

        # Initialize with small weights (matches Phase 2)
        for module in [self.v_proj, self.t_proj]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self.norm = nn.LayerNorm(768)

        # T5 projection
        self.to_t5 = nn.Linear(768, 768)
        nn.init.xavier_uniform_(self.to_t5.weight, gain=0.1)
        nn.init.zeros_(self.to_t5.bias)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(768, 384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(384, num_classes)
        )

    def get_fused_features(self, pixel_values, input_ids, attention_mask):
        """Get fused vision-language features"""
        # Vision encoding
        with torch.no_grad():
            v = self.vit(pixel_values).last_hidden_state
        v = torch.clamp(v, min=-10, max=10)
        v = self.v_proj(v)

        # Text encoding
        with torch.no_grad():
            t = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        t = torch.clamp(t, min=-10, max=10)
        t = self.t_proj(t)

        # Cross-modal fusion
        attn_out, _ = self.cross_attn(v, t, t)
        fused = self.norm(v + attn_out)

        return fused

    def classify(self, pixel_values, input_ids, attention_mask):
        """Classification forward pass"""
        self.eval()
        with torch.no_grad():
            fused = self.get_fused_features(pixel_values, input_ids, attention_mask)
            # Use CLS token (first token) for classification
            logits = self.cls_head(fused[:, 0])
            probs = F.softmax(logits, dim=-1)
        return logits, probs

    def generate(self, pixel_values, input_ids, attention_mask, max_length=128):
        """VQA generation forward pass"""
        self.eval()
        with torch.no_grad():
            fused = self.get_fused_features(pixel_values, input_ids, attention_mask)

            # Project to T5 space
            h = self.to_t5(fused)
            h = F.layer_norm(h, h.shape[-1:])

            encoder_outputs = BaseModelOutput(last_hidden_state=h)

            return self.t5.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=torch.ones(h.shape[:2], device=h.device),
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
