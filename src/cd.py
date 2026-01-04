"""
Contrastive Decoding module for SAE-PD.
Implements Distributional Value Decoding (DVD) and Contrastive-Adaptive Decoding (CAD).
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score

from utils import (
    clip_decode, prob_map, dvd_decoding, LABEL_MAPPINGS, IND_TO_OPTION_MAP,
    evaluate
)

logger = logging.getLogger(__name__)


# =============================================================================
# Contrastive Decoding Functions
# =============================================================================

def cad_decode(
    logits: torch.Tensor,
    base_logits: torch.Tensor,
    alpha: float = 0.5,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Contrastive-Adaptive Decoding (CAD).
    
    Contrasts conditional logits against baseline to emphasize differences.
    
    Args:
        logits: Conditional logits
        base_logits: Baseline (unconditional) logits
        alpha: Contrastive weight (higher = more contrast)
        temperature: Temperature for softmax
    
    Returns:
        Contrasted logits
    """
    # Apply temperature
    logits_temp = logits / temperature
    base_temp = base_logits / temperature
    
    # Contrastive combination: (1 + alpha) * p_cond - alpha * p_base
    contrasted = (1 + alpha) * logits_temp - alpha * base_temp
    
    # Mask out very unlikely tokens
    contrasted = torch.where(base_logits > -100, contrasted, logits)
    
    return contrasted


def dvd_decode_multi(
    conditional_logits: List[torch.Tensor],
    unconditional_logits: torch.Tensor,
    tau: float = 0.4,
    topk: int = 10,
    alpha: float = 0.2,
    weighting: str = 'entropy'
) -> torch.Tensor:
    """
    Distributional Value Decoding with multiple conditional distributions.
    
    Combines multiple conditional logits using entropy-weighted averaging,
    then contrasts against unconditional baseline.
    
    Args:
        conditional_logits: List of conditional logits from different sources
        unconditional_logits: Baseline unconditional logits
        tau: Temperature for softmax
        topk: Number of top tokens for entropy calculation
        alpha: Contrastive weight
        weighting: Weighting strategy ('entropy', 'uniform')
    
    Returns:
        Combined logits
    """
    if len(conditional_logits) == 0:
        return unconditional_logits
    
    if len(conditional_logits) == 1:
        return cad_decode(conditional_logits[0], unconditional_logits, alpha)
    
    # Compute entropy for each conditional distribution
    entropies = []
    contrasted_logits = []
    
    for cond_logits in conditional_logits:
        # Compute entropy
        probas = torch.nn.functional.softmax(cond_logits / tau, dim=-1)
        values, indices = torch.topk(probas, topk, largest=True)
        V = cond_logits[..., indices] if cond_logits.dim() == 1 else cond_logits[indices]
        entropy = -(V.exp() * V.clip(-100, 0)).sum(dim=-1)
        if entropy.dim() > 0:
            entropy = entropy.item() if entropy.numel() == 1 else entropy.mean().item()
        entropies.append(entropy)
        
        # Apply contrastive decoding
        contrasted = cad_decode(cond_logits, unconditional_logits, alpha)
        contrasted_logits.append(contrasted)
    
    # Combine based on weighting strategy
    if weighting == 'entropy':
        # Higher entropy = more uncertainty = lower weight
        weights = torch.nn.functional.softmax(
            torch.tensor(entropies, dtype=torch.float32), dim=0
        )
    else:  # uniform
        weights = torch.ones(len(conditional_logits)) / len(conditional_logits)
    
    # Weighted combination
    stacked = torch.stack(contrasted_logits, dim=0)
    if stacked.dim() == 2:  # (num_cond, vocab)
        combined = torch.matmul(weights.unsqueeze(0), stacked).squeeze(0)
    else:  # (num_cond, batch, vocab)
        combined = torch.einsum('i,ijk->jk', weights, stacked)
    
    return combined


# =============================================================================
# DVD Evaluation
# =============================================================================

def run_dvd(
    model: HookedTransformer,
    test_data: Union[Dataset, List[Dict]],
    train_data: Union[Dataset, List[Dict]],
    dataset: str,
    num_comments: int = 3,
    tau: float = 0.4,
    alpha: float = 0.1,
    topk: int = 20,
    output_dir: Optional[str] = None
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Run Distributional Value Decoding evaluation.
    
    Args:
        model: HookedTransformer model
        test_data: Test dataset
        train_data: Training dataset (for comment pool)
        dataset: Dataset name
        num_comments: Number of comment sources
        tau: Temperature for DVD
        alpha: Contrastive weight
        topk: Top-k for entropy calculation
        output_dir: Output directory
    
    Returns:
        Tuple of (results list, metrics dataframe)
    """
    # Convert to list if needed
    if isinstance(test_data, Dataset):
        test_list = [test_data[i] for i in range(len(test_data))]
    else:
        test_list = list(test_data)
    
    if isinstance(train_data, Dataset):
        train_list = [train_data[i] for i in range(len(train_data))]
    else:
        train_list = list(train_data)
    
    # Determine options
    if dataset == 'gqa':
        options = test_list[0].get('options', ['A', 'B', 'C', 'D', 'E'])
    else:
        options = ['No', 'Yes', 'Unsure']
    
    # First, compute baseline logits
    logger.info("Computing baseline logits")
    baseline_results = []
    
    for item in tqdm(test_list, desc="Baseline"):
        prompt = format_dvd_prompt(item, dataset)
        tokenized = model.to_tokens(prompt)
        
        logits = model.forward(
            tokenized,
            return_type="logits",
            prepend_bos=True,
            padding_side='left',
        ).detach().cpu()
        
        baseline_results.append({
            'logits': logits[0, -1, :],
            'decoded': clip_decode(logits[:, -1], model.tokenizer),
            'probs': prob_map(logits, options, model.tokenizer)
        })
    
    # Compute DVD results
    logger.info("Computing DVD results")
    dvd_results = []
    
    for i, item in enumerate(tqdm(test_list, desc="DVD")):
        # Get conditional logits for each comment
        conditional_logits = []
        
        comments = item.get('comments', [''] * num_comments)
        for c_idx in range(min(num_comments, len(comments))):
            prompt = format_dvd_prompt(item, dataset, comment=comments[c_idx])
            tokenized = model.to_tokens(prompt)
            
            logits = model.forward(
                tokenized,
                return_type="logits",
                prepend_bos=True,
                padding_side='left',
            ).detach().cpu()
            
            conditional_logits.append(logits[0, -1, :])
        
        # Apply DVD
        base = baseline_results[i]['logits']
        final_logits = dvd_decode_multi(
            conditional_logits, base,
            tau=tau, alpha=alpha, topk=topk
        )
        
        decoded = clip_decode(final_logits.unsqueeze(0), model.tokenizer)
        probs = prob_map(final_logits.unsqueeze(0), options, model.tokenizer)
        
        dvd_results.append({
            'logits': final_logits,
            'decoded': decoded,
            'probs': probs,
            'conditional_logits': conditional_logits,
            'gold': item.get('gold_distribution', item.get('label', item.get('legal_majority')))
        })
    
    model.reset_hooks()
    
    # Compute metrics
    metrics = compute_dvd_metrics(baseline_results, dvd_results, test_list, dataset, model.tokenizer)
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create results dataframe
        df = create_dvd_dataframe(baseline_results, dvd_results, test_list, dataset, model.tokenizer)
        df.to_csv(os.path.join(output_dir, 'dvd_results.csv'), index=False)
        
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(os.path.join(output_dir, 'dvd_metrics.csv'), index=False)
        
        logger.info(f"Saved DVD results to {output_dir}")
    
    return dvd_results, pd.DataFrame(metrics)


def format_dvd_prompt(
    item: Dict,
    dataset: str,
    comment: Optional[str] = None,
    legal_prompt: str = ''
) -> str:
    """
    Format prompt for DVD evaluation.
    
    Args:
        item: Data item
        dataset: Dataset name
        comment: Optional comment to include
        legal_prompt: Legal definition prompt
    
    Returns:
        Formatted prompt
    """
    if dataset == 'gqa':
        if comment:
            return f"You are from the country of {item.get('attribute', '')}, respond to the following instruction with the help of a passage.\n\nPassage: {comment}\n\n{item.get('question', item.get('text', ''))} "
        else:
            return f"You are from the country of {item.get('attribute', '')}, respond to the following instruction.\n\n{item.get('question', item.get('text', ''))} "
    
    elif dataset == 'mislc':
        base = f"Claim: {item.get('claim', '')}\n\nDoes this claim contain misinformation? Answer Yes, No, or Unsure."
        if comment:
            return f"{legal_prompt}\n\n{base}\n\nThinking: {comment}\nAnswer: <strong>"
        else:
            return f"{legal_prompt}\n\n{base}\nAnswer: <strong>"
    
    elif dataset == 'hatespeech':
        base = f"Post: {item.get('text', '')}\n\nDoes this post violate the above hate speech policy? Answer Yes, No, or Unsure."
        if comment:
            return f"{base}\n\nThinking: {comment}\nAnswer: <strong>"
        else:
            return f"{base}\nAnswer: <strong>"
    
    return str(item)


def compute_dvd_metrics(
    baseline_results: List[Dict],
    dvd_results: List[Dict],
    test_data: List[Dict],
    dataset: str,
    tokenizer: AutoTokenizer
) -> Dict[str, List]:
    """
    Compute metrics for DVD evaluation.
    
    Args:
        baseline_results: Baseline results
        dvd_results: DVD results
        test_data: Test data items
        dataset: Dataset name
        tokenizer: Tokenizer
    
    Returns:
        Metrics dictionary
    """
    mapping = LABEL_MAPPINGS.get(dataset, {})
    
    metrics = {
        'setting': ['baseline', 'dvd'],
        'macro-f1': [],
        'micro-f1': [],
        'weighted-f1': [],
    }
    
    if dataset != 'gqa':
        metrics['precision-1'] = []
        metrics['recall-1'] = []
        metrics['f1-1'] = []
    
    for setting, results in [('baseline', baseline_results), ('dvd', dvd_results)]:
        predictions = []
        gold_labels = []
        
        for i, (result, item) in enumerate(zip(results, test_data)):
            pred_token = tokenizer.decode(result['decoded'][1][0, 0].item()).strip().lower()
            
            if dataset == 'gqa':
                pred = LABEL_MAPPINGS.get('gqa', {}).get(pred_token.upper(), -1)
                gold = np.argmax(item.get('gold_distribution', [0])) if isinstance(item.get('gold_distribution'), (list, np.ndarray)) else -1
            else:
                pred = mapping.get(pred_token, -1)
                if pred == -1:
                    pred = 0  # Default to 'No'
                gold = item.get('label', item.get('legal_majority', 0))
                if isinstance(gold, str):
                    gold = mapping.get(gold.lower(), 0)
            
            predictions.append(0 if pred < 0 else pred)
            gold_labels.append(gold)
        
        metrics['macro-f1'].append(f1_score(gold_labels, predictions, average='macro', zero_division=0))
        metrics['micro-f1'].append(f1_score(gold_labels, predictions, average='micro', zero_division=0))
        metrics['weighted-f1'].append(f1_score(gold_labels, predictions, average='weighted', zero_division=0))
        
        if dataset != 'gqa':
            metrics['precision-1'].append(precision_score(gold_labels, predictions, labels=[1], average='macro', zero_division=0))
            metrics['recall-1'].append(recall_score(gold_labels, predictions, labels=[1], average='micro', zero_division=0))
            metrics['f1-1'].append(f1_score(gold_labels, predictions, labels=[1], average='macro', zero_division=0))
    
    logger.info(f"DVD metrics: {metrics}")
    return metrics


def create_dvd_dataframe(
    baseline_results: List[Dict],
    dvd_results: List[Dict],
    test_data: List[Dict],
    dataset: str,
    tokenizer: AutoTokenizer
) -> pd.DataFrame:
    """
    Create results dataframe for DVD evaluation.
    
    Args:
        baseline_results: Baseline results
        dvd_results: DVD results
        test_data: Test data items
        dataset: Dataset name
        tokenizer: Tokenizer
    
    Returns:
        Results dataframe
    """
    mapping = LABEL_MAPPINGS.get(dataset, {})
    
    data = {
        'baseline_pred': [tokenizer.decode(r['decoded'][1][0, 0].item()).strip() for r in baseline_results],
        'dvd_pred': [tokenizer.decode(r['decoded'][1][0, 0].item()).strip() for r in dvd_results],
        'baseline_probs': [r['probs'] for r in baseline_results],
        'dvd_probs': [r['probs'] for r in dvd_results],
    }
    
    if dataset == 'gqa':
        data['gold_dist'] = [item.get('gold_distribution') for item in test_data]
        data['gold_label'] = [np.argmax(item.get('gold_distribution', [0])) for item in test_data]
    else:
        data['gold_label'] = [item.get('label', item.get('legal_majority', 0)) for item in test_data]
    
    return pd.DataFrame(data)
