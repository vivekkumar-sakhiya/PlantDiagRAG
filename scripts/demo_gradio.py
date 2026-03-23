"""
PlantDiagRAG Gradio Demo

A web interface for plant disease diagnosis using the PlantDiagRAG pipeline.

Usage:
    python demo_gradio.py --vqa_checkpoint path/to/vqa_model.pt \
                          --cls_checkpoint path/to/classifier.pt \
                          --knowledge_base path/to/kb.json \
                          --label_mapping path/to/labels.json

Or with default paths:
    python demo_gradio.py
"""

import argparse
import gradio as gr
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plantdiagrag import PlantDiagRAGPipeline


def create_demo(pipeline):
    """Create the Gradio interface."""
    
    def diagnose(image, question):
        """Gradio interface function."""
        if image is None:
            return "Please upload an image.", "", ""
        
        # Run diagnosis
        result = pipeline.diagnose(image, question=question if question else None)
        
        # Format classification output
        cls_out = f"""**Predicted Disease:** {result['classification']['predicted_class']}
**Confidence:** {result['classification']['confidence']*100:.1f}%

**Top 3 Predictions:**
"""
        for i, (cls, prob) in enumerate(result['classification']['top3'], 1):
            cls_out += f"{i}. {cls}: {prob*100:.1f}%\n"
        
        if result['classification']['is_healthy']:
            cls_out += "\n✅ **This plant appears to be healthy!**"
        
        # Format VQA output
        vqa_out = f"""**Question:** {result['vqa']['question']}

**Answer:** {result['vqa']['answer']}
"""
        
        # Treatment output
        treatment_out = result['treatment']['summary']
        if result['treatment']['sources']:
            treatment_out += f"\n\n**Sources:** {', '.join(result['treatment']['sources'])}"
        
        return cls_out, vqa_out, treatment_out
    
    # Create interface
    with gr.Blocks(title="PlantDiagRAG", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# 🌱 PlantDiagRAG: Plant Disease Diagnosis System

Upload an image of a plant leaf to get:
1. **Disease Classification** - Identify the disease with confidence scores
2. **Visual Q&A** - Get answers to questions about the plant
3. **Treatment Recommendations** - Evidence-based treatment from agricultural databases

---
""")
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Plant Image")
                question_input = gr.Textbox(
                    label="Question (Optional)",
                    placeholder="e.g., What are the visible symptoms?",
                    lines=2
                )
                submit_btn = gr.Button("🔬 Diagnose", variant="primary")
            
            with gr.Column(scale=2):
                with gr.Tab("🏷️ Classification"):
                    cls_output = gr.Markdown()
                
                with gr.Tab("❓ Visual Q&A"):
                    vqa_output = gr.Markdown()
                
                with gr.Tab("💊 Treatment"):
                    treatment_output = gr.Markdown()
        
        # Example images
        gr.Examples(
            examples=[
                ["examples/tomato_late_blight.jpg", "What disease is affecting this plant?"],
                ["examples/apple_scab.jpg", "How can I treat this disease?"],
                ["examples/healthy_leaf.jpg", "Is this plant healthy?"],
            ],
            inputs=[image_input, question_input],
            label="Example Images"
        )
        
        gr.Markdown("""
---
### About PlantDiagRAG

- **38 Disease Classes** from PlantVillage dataset
- **99.10% Classification Accuracy**
- **Knowledge Base**: ICAR, UC IPM, PNW Handbook, AGROVOC (54 documents)
- **Architecture**: ViT-Base + BERT-Base + Flan-T5-Base with LoRA

📄 [Paper](#) | 💻 [GitHub](https://github.com/vivekkumar-sakhiya/PlantDiagRAG)
""")
        
        submit_btn.click(
            fn=diagnose,
            inputs=[image_input, question_input],
            outputs=[cls_output, vqa_output, treatment_output]
        )
    
    return demo


def main():
    parser = argparse.ArgumentParser(description="PlantDiagRAG Gradio Demo")
    parser.add_argument("--vqa_checkpoint", type=str, 
                        default="checkpoints/best_vqa_model.pt",
                        help="Path to VQA model checkpoint")
    parser.add_argument("--cls_checkpoint", type=str,
                        default="checkpoints/best_classifier_v2.pt", 
                        help="Path to classifier checkpoint")
    parser.add_argument("--knowledge_base", type=str,
                        default="knowledge_base/all_kb_documents.json",
                        help="Path to knowledge base JSON")
    parser.add_argument("--label_mapping", type=str,
                        default="configs/label_mapping.json",
                        help="Path to label mapping JSON")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--share", action="store_true",
                        help="Create public share link")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port to run on")
    
    args = parser.parse_args()
    
    print("🌱 Loading PlantDiagRAG pipeline...")
    pipeline = PlantDiagRAGPipeline.from_pretrained(
        vqa_checkpoint=args.vqa_checkpoint,
        classifier_checkpoint=args.cls_checkpoint,
        knowledge_base=args.knowledge_base,
        label_mapping_path=args.label_mapping,
        device=args.device
    )
    
    print("🚀 Starting Gradio demo...")
    demo = create_demo(pipeline)
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
