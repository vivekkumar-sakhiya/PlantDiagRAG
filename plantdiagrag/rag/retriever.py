"""
RAG (Retrieval-Augmented Generation) module for plant disease treatment recommendations.

Uses FAISS for semantic search over agricultural knowledge base documents
from ICAR, UC IPM, PNW Handbook, and AGROVOC sources.
"""

import re
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class PlantDiseaseRAG:
    """
    RAG system for retrieving plant disease treatment information.
    """
    
    def __init__(self, knowledge_base_path, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the RAG system.
        
        Args:
            knowledge_base_path: Path to the knowledge base JSON file
            model_name: Sentence transformer model name
        """
        self.sentence_model = SentenceTransformer(model_name)
        self.doc_metadata = []
        self.index = None
        
        # Load knowledge base
        self._load_knowledge_base(knowledge_base_path)
        self._build_index()
    
    def _load_knowledge_base(self, path):
        """Load knowledge base documents."""
        with open(path, 'r') as f:
            self.doc_metadata = json.load(f)
        print(f"✓ Loaded {len(self.doc_metadata)} documents")
    
    def _build_index(self):
        """Build FAISS index from documents."""
        # Create embeddings for all documents
        texts = [f"{doc['title']} {doc['content']}" for doc in self.doc_metadata]
        embeddings = self.sentence_model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create index
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        print(f"✓ Built FAISS index with {self.index.ntotal} vectors")
    
    def get_treatment_info(self, query, disease_class=None, k=3):
        """
        Retrieve treatment information for a plant disease.
        
        Args:
            query: Search query
            disease_class: Disease class name (e.g., "Tomato___Late_blight")
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with treatment information
        """
        results = {
            'disease': disease_class,
            'documents': [],
            'treatment_summary': '',
            'sources': []
        }

        # Parse disease class
        plant = ""
        disease = ""
        if disease_class and '___' in disease_class:
            parts = disease_class.split('___')
            plant = parts[0].replace('_', ' ').replace('(', '').replace(')', '').strip()
            disease = parts[1].replace('_', ' ').strip()

        # Check if healthy
        if disease_class and 'healthy' in disease_class.lower():
            results['treatment_summary'] = f"""✅ **HEALTHY PLANT DETECTED**

This {plant} appears to be healthy with no visible signs of disease.

**Recommendations:**
- Continue regular watering and fertilization schedule
- Monitor for any changes in leaf color or texture
- Ensure proper spacing for air circulation
- Practice preventive care with regular inspections
"""
            results['sources'] = ['General Best Practices']
            return results

        # Create strict disease matching pattern
        disease_lower = disease.lower()
        disease_words = [w for w in disease_lower.split() if len(w) > 2]

        # Search documents
        enhanced_query = f"{plant} {disease} treatment control"
        query_embedding = self.sentence_model.encode([enhanced_query])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, 20)

        # Filter documents strictly
        relevant_docs = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.doc_metadata):
                continue

            doc = self.doc_metadata[idx]
            content_lower = doc['content'].lower()

            # STRICT CHECK: The exact disease name or all key words must appear together
            has_exact_disease = disease_lower in content_lower

            # Check if all disease words appear within 50 chars of each other
            has_disease_words = False
            if len(disease_words) >= 2:
                first_word = disease_words[0]
                if first_word in content_lower:
                    idx_first = content_lower.find(first_word)
                    nearby_text = content_lower[max(0, idx_first-30):idx_first+100]
                    has_disease_words = all(w in nearby_text for w in disease_words)
            elif len(disease_words) == 1:
                has_disease_words = disease_words[0] in content_lower

            if not (has_exact_disease or has_disease_words):
                continue

            # Skip photo-heavy documents
            if content_lower.count('photo') > 3:
                continue

            # Skip short AGROVOC entries
            if doc['source'] == 'AGROVOC' and len(doc['content']) < 100:
                continue

            # Check plant relevance
            plant_lower = plant.lower()
            plant_words = [w for w in plant_lower.split() if len(w) > 2]
            has_plant = any(w in content_lower or w in doc['title'].lower() for w in plant_words)

            # Score - heavily prioritize ICAR for actionable guidelines
            relevance = float(score)
            if has_plant:
                relevance += 1.0
            if has_exact_disease:
                relevance += 0.5
            if 'ICAR' in doc['source']:
                relevance += 2.0  # Strong ICAR preference
            elif 'UC IPM' in doc['source']:
                relevance += 0.5

            relevant_docs.append({
                'title': doc['title'],
                'content': doc['content'],
                'source': doc['source'],
                'score': relevance
            })

        # Sort and take top k
        relevant_docs.sort(key=lambda x: x['score'], reverse=True)
        results['documents'] = relevant_docs[:k]

        # Build summary
        if results['documents']:
            summary_parts = [f"🔬 **DIAGNOSIS: {disease.title()}** on {plant.title()}\n"]
            sources = set()

            for i, doc in enumerate(results['documents'], 1):
                sources.add(doc['source'])

                # Extract disease-specific section
                content = doc['content']
                content_lower = content.lower()

                section = None
                for kw in disease_words:
                    if kw in content_lower:
                        idx = content_lower.find(kw)
                        start = content_lower.rfind('\n', 0, idx)
                        start = 0 if start == -1 else start + 1

                        # Find end
                        remaining = content[idx+len(kw):]
                        match = re.search(r'\n[A-Z][A-Z\s]{3,}[(:)]', remaining)
                        end = idx + len(kw) + match.start() if match else min(idx + 500, len(content))

                        section = content[start:end].strip()
                        lines = [l for l in section.split('\n') if 'photo' not in l.lower()]
                        section = '\n'.join(lines)

                        if len(section) > 450:
                            section = section[:450] + "..."
                        break

                if not section:
                    section = content[:400] + "..." if len(content) > 400 else content

                summary_parts.append(f"**{i}. {doc['title']}** ({doc['source']}):\n{section}\n")

            results['treatment_summary'] = "\n".join(summary_parts)
            results['sources'] = list(sources)
        else:
            results['treatment_summary'] = f"""⚠️ **No Specific Information Found**

Disease: {disease.title()} on {plant.title()}

**General Recommendations:**
- Remove and destroy infected plant parts
- Improve air circulation
- Avoid overhead watering
- Apply appropriate fungicide (consult local extension)
- Practice crop rotation

Consult your local agricultural extension for specific treatment.
"""
            results['sources'] = ['General Recommendations']

        return results
