"""
PlantDiagRAG Inference Script

Diagnose plant diseases from command line.

Usage:
    python inference.py --image path/to/image.jpg
    python inference.py --image path/to/image.jpg --question "What disease is this?"
    python inference.py --image_dir path/to/images/ --output results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plantdiagrag import PlantDiagRAGPipeline


def diagnose_single(pipeline, image_path, question=None, verbose=True):
    """Diagnose a single image."""
    result = pipeline.diagnose(image_path, question=question)
    
    if verbose:
        print("\n" + "="*60)
        print("DIAGNOSIS RESULTS")
        print("="*60)
        
        # Classification
        cls = result['classification']
        print(f"\n🏷️  CLASSIFICATION")
        print(f"   Disease: {cls['predicted_class']}")
        print(f"   Confidence: {cls['confidence']*100:.1f}%")
        print(f"   Top 3:")
        for i, (name, prob) in enumerate(cls['top3'], 1):
            print(f"      {i}. {name}: {prob*100:.1f}%")
        
        # VQA
        vqa = result['vqa']
        print(f"\n❓ VISUAL Q&A")
        print(f"   Q: {vqa['question']}")
        print(f"   A: {vqa['answer']}")
        
        # Treatment
        print(f"\n💊 TREATMENT RECOMMENDATIONS")
        print(f"   Sources: {', '.join(result['treatment']['sources'])}")
        print(f"\n{result['treatment']['summary']}")
        
        print("\n" + "="*60)
    
    return result


def diagnose_batch(pipeline, image_dir, output_path, question=None):
    """Diagnose all images in a directory."""
    image_dir = Path(image_dir)
    results = []
    
    # Find all images
    extensions = ['.jpg', '.jpeg', '.png', '.webp']
    images = [f for f in image_dir.iterdir() 
              if f.suffix.lower() in extensions]
    
    print(f"Found {len(images)} images in {image_dir}")
    
    for i, img_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] Processing: {img_path.name}")
        
        try:
            result = pipeline.diagnose(str(img_path), question=question)
            result['image'] = str(img_path)
            results.append(result)
            
            cls = result['classification']
            print(f"   → {cls['predicted_class']} ({cls['confidence']*100:.1f}%)")
        except Exception as e:
            print(f"   ✗ Error: {e}")
            results.append({
                'image': str(img_path),
                'error': str(e)
            })
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="PlantDiagRAG Inference")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Path to single image")
    input_group.add_argument("--image_dir", type=str, help="Directory of images for batch processing")
    
    # Model paths
    parser.add_argument("--vqa_checkpoint", type=str, 
                        default="checkpoints/best_vqa_model.pt")
    parser.add_argument("--cls_checkpoint", type=str,
                        default="checkpoints/best_classifier_v2.pt")
    parser.add_argument("--knowledge_base", type=str,
                        default="knowledge_base/all_kb_documents.json")
    parser.add_argument("--label_mapping", type=str,
                        default="configs/label_mapping.json")
    
    # Options
    parser.add_argument("--question", type=str, default=None,
                        help="Question for VQA (optional)")
    parser.add_argument("--output", type=str, default="results.json",
                        help="Output file for batch processing")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON (for single image)")
    
    args = parser.parse_args()
    
    # Load pipeline
    print("🌱 Loading PlantDiagRAG pipeline...")
    pipeline = PlantDiagRAGPipeline.from_pretrained(
        vqa_checkpoint=args.vqa_checkpoint,
        classifier_checkpoint=args.cls_checkpoint,
        knowledge_base=args.knowledge_base,
        label_mapping_path=args.label_mapping,
        device=args.device
    )
    
    # Run inference
    if args.image:
        result = diagnose_single(
            pipeline, 
            args.image, 
            question=args.question,
            verbose=not args.json
        )
        
        if args.json:
            print(json.dumps(result, indent=2))
    else:
        diagnose_batch(
            pipeline,
            args.image_dir,
            args.output,
            question=args.question
        )


if __name__ == "__main__":
    main()
