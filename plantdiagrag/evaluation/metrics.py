"""
Evaluation metrics for PlantDiagRAG.

Includes metrics for:
- VQA: BLEU, ROUGE, METEOR, Exact Match, Token F1
- Classification: Accuracy, Precision, Recall, F1
"""

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Initialize ROUGE scorer
_rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
_smooth = SmoothingFunction().method1


def calc_bleu(reference, hypothesis):
    """
    Calculate BLEU score between reference and hypothesis.
    
    Args:
        reference: Ground truth text
        hypothesis: Generated text
        
    Returns:
        BLEU score (0-1)
    """
    try:
        ref_tokens = word_tokenize(reference.lower())
        hyp_tokens = word_tokenize(hypothesis.lower())
        if len(hyp_tokens) == 0:
            return 0.0
        return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=_smooth)
    except:
        return 0.0


def calc_rouge(reference, hypothesis):
    """
    Calculate ROUGE scores between reference and hypothesis.
    
    Args:
        reference: Ground truth text
        hypothesis: Generated text
        
    Returns:
        Dictionary with rouge1, rouge2, rougeL F-measures
    """
    try:
        scores = _rouge_scorer.score(reference, hypothesis)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    except:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}


def calc_meteor(reference, hypothesis):
    """
    Calculate METEOR score between reference and hypothesis.
    
    Args:
        reference: Ground truth text
        hypothesis: Generated text
        
    Returns:
        METEOR score (0-1)
    """
    try:
        ref_tokens = word_tokenize(reference.lower())
        hyp_tokens = word_tokenize(hypothesis.lower())
        return meteor_score([ref_tokens], hyp_tokens)
    except:
        return 0.0


def exact_match(reference, hypothesis):
    """
    Check if reference and hypothesis match exactly (normalized).
    
    Args:
        reference: Ground truth text
        hypothesis: Generated text
        
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    return float(reference.strip().lower() == hypothesis.strip().lower())


def f1_token(reference, hypothesis):
    """
    Calculate token-level F1 score between reference and hypothesis.
    
    Args:
        reference: Ground truth text
        hypothesis: Generated text
        
    Returns:
        Token F1 score (0-1)
    """
    try:
        ref_tokens = set(word_tokenize(reference.lower()))
        hyp_tokens = set(word_tokenize(hypothesis.lower()))

        if len(hyp_tokens) == 0:
            return 0.0
        if len(ref_tokens) == 0:
            return 0.0

        common = ref_tokens & hyp_tokens
        precision = len(common) / len(hyp_tokens)
        recall = len(common) / len(ref_tokens)

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    except:
        return 0.0


def evaluate_vqa_predictions(references, hypotheses):
    """
    Evaluate VQA predictions against references.
    
    Args:
        references: List of ground truth answers
        hypotheses: List of generated answers
        
    Returns:
        Dictionary with all metrics
    """
    assert len(references) == len(hypotheses), "References and hypotheses must have same length"
    
    results = {
        'bleu': [],
        'rouge1': [],
        'rouge2': [],
        'rougeL': [],
        'meteor': [],
        'exact_match': [],
        'f1_token': []
    }
    
    for ref, hyp in zip(references, hypotheses):
        results['bleu'].append(calc_bleu(ref, hyp))
        
        rouge_scores = calc_rouge(ref, hyp)
        results['rouge1'].append(rouge_scores['rouge1'])
        results['rouge2'].append(rouge_scores['rouge2'])
        results['rougeL'].append(rouge_scores['rougeL'])
        
        results['meteor'].append(calc_meteor(ref, hyp))
        results['exact_match'].append(exact_match(ref, hyp))
        results['f1_token'].append(f1_token(ref, hyp))
    
    # Calculate averages
    summary = {
        'bleu': sum(results['bleu']) / len(results['bleu']),
        'rouge1': sum(results['rouge1']) / len(results['rouge1']),
        'rouge2': sum(results['rouge2']) / len(results['rouge2']),
        'rougeL': sum(results['rougeL']) / len(results['rougeL']),
        'meteor': sum(results['meteor']) / len(results['meteor']),
        'exact_match': sum(results['exact_match']) / len(results['exact_match']),
        'f1_token': sum(results['f1_token']) / len(results['f1_token']),
        'num_samples': len(references)
    }
    
    return summary, results


def evaluate_classification(y_true, y_pred, average='weighted'):
    """
    Evaluate classification predictions.
    
    Args:
        y_true: List of true labels
        y_pred: List of predicted labels
        average: Averaging method for multi-class metrics
        
    Returns:
        Dictionary with classification metrics
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
        'num_samples': len(y_true)
    }
