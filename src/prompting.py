"""
Prompting module for SAE-PD.
Handles zero-shot and few-shot prompting for different datasets.
"""

import os
import json
import random
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
    clip_decode, prob_map, LABEL_MAPPINGS, IND_TO_OPTION_MAP
)

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt Templates
# =============================================================================

TEMPLATES = {
    'gqa': {
        'zero_shot': "You are from the country of {attribute}, respond to the following instruction.\n\n{question} ",
        'few_shot': "You are from the country of {attribute}, respond to the following instruction with the help of a passage.\n\nPassage: {comment}\n\n{question} {answer}"
    },
    'mislc': {
        'zero_shot': "{legal_prompt}\nClaim: {claim}\n\nDoes this claim contain misinformation? Answer Yes, No, or Unsure.\n\nAnswer: ",
        'few_shot': "{legal_prompt}\nClaim: {claim}\n\nDoes this claim contain misinformation? Answer Yes, No, or Unsure.\n\nThinking: {comment}\n\nAnswer: {answer}"
    },
    'hatespeech': {
        'zero_shot': "Post: {text}\n\nDoes this post violate the above hate speech policy? Answer Yes, No, or Unsure.\n\nAnswer: ",
        'few_shot': "Post: {text}\n\nDoes this post violate the above hate speech policy? Answer Yes, No, or Unsure.\n\nThinking: {comment}\n\nAnswer: {answer}"
    }
}

LABEL_TO_ANSWER = {
    'mislc': {0: 'No', 1: 'Yes', 2: 'Unsure'},
    'hatespeech': {'Does not violate': 'No', 'Violates': 'Yes', 'Meaning unclear': 'Unsure',
                   0: 'No', 1: 'Yes', 2: 'Unsure'}
}


# =============================================================================
# Prompting Functions
# =============================================================================

def format_prompt(
    item: Dict,
    dataset: str,
    few_shot_examples: Optional[List[Dict]] = None,
    template_type: str = 'zero_shot',
    legal_prompt: str = '',
    definitions: Optional[List[str]] = None
) -> str:
    """
    Format a prompt for the given dataset and item.
    
    Args:
        item: Data item with required fields
        dataset: Dataset name ('gqa', 'mislc', 'hatespeech')
        few_shot_examples: Optional list of few-shot examples
        template_type: 'zero_shot' or 'few_shot'
        legal_prompt: Legal definition prompt for mislc
        definitions: Definition prompts for hatespeech
    
    Returns:
        Formatted prompt string
    """
    templates = TEMPLATES.get(dataset, TEMPLATES['gqa'])
    
    if dataset == 'gqa':
        if few_shot_examples and len(few_shot_examples) > 0:
            # Build few-shot prompt
            examples = []
            for ex in few_shot_examples:
                comment = ex.get('comments', [''])[0] if ex.get('comments') else ''
                answer = ex.get('rand_ans', [''])[0] if ex.get('rand_ans') else ''
                examples.append(templates['few_shot'].format(
                    attribute=ex.get('attribute', ''),
                    question=ex.get('question', ex.get('text', '')),
                    comment=comment,
                    answer=answer
                ))
            
            # Add test question without answer
            prompt = '\n\n'.join(examples) + '\n\n' + templates['zero_shot'].format(
                attribute=item.get('attribute', ''),
                question=item.get('question', item.get('text', ''))
            )
        else:
            prompt = templates['zero_shot'].format(
                attribute=item.get('attribute', ''),
                question=item.get('question', item.get('text', ''))
            )
    
    elif dataset == 'mislc':
        label_map = LABEL_TO_ANSWER.get('mislc', {})
        
        if few_shot_examples and len(few_shot_examples) > 0:
            examples = []
            for ex in few_shot_examples:
                comment = ex.get('comments', [''])[0] if ex.get('comments') else ''
                label = ex.get('legal_majority', 0)
                answer = label_map.get(label, 'Unsure')
                examples.append(templates['few_shot'].format(
                    legal_prompt='',
                    claim=ex.get('claim', ''),
                    comment=comment,
                    answer=answer
                ))
            
            prompt = '\n\n'.join(examples) + '\n\n' + templates['zero_shot'].format(
                legal_prompt=legal_prompt,
                claim=item.get('claim', '')
            )
        else:
            prompt = templates['zero_shot'].format(
                legal_prompt=legal_prompt,
                claim=item.get('claim', '')
            )
    
    elif dataset == 'hatespeech':
        label_map = LABEL_TO_ANSWER.get('hatespeech', {})
        
        # Add definition if available
        def_text = definitions[0][:400] if definitions else ''
        if def_text:
            prefix = f"hate speech policy: {def_text}\n\n"
        else:
            prefix = ''
        
        if few_shot_examples and len(few_shot_examples) > 0:
            examples = []
            for ex in few_shot_examples:
                comment = ex.get('comments', [''])[0] if ex.get('comments') else ''
                label = ex.get('label', 'Does not violate')
                answer = label_map.get(label, 'No')
                examples.append(templates['few_shot'].format(
                    text=ex.get('text', ''),
                    comment=comment,
                    answer=answer
                ))
            
            prompt = prefix + '\n\n'.join(examples) + '\n\n' + templates['zero_shot'].format(
                text=item.get('text', '')
            )
        else:
            prompt = prefix + templates['zero_shot'].format(
                text=item.get('text', '')
            )
    
    else:
        # Default fallback
        prompt = str(item)
    
    return prompt


def run_prompting(
    model: HookedTransformer,
    test_data: Union[Dataset, List[Dict]],
    train_data: Optional[Union[Dataset, List[Dict]]] = None,
    dataset: str = 'gqa',
    shots: int = 0,
    legal_prompt: str = '',
    definitions: Optional[List[str]] = None,
    max_seq_len: int = 1200,
    output_dir: Optional[str] = None
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Run prompting evaluation on test data.
    
    Args:
        model: HookedTransformer model
        test_data: Test dataset
        train_data: Training dataset for few-shot examples
        dataset: Dataset name
        shots: Number of few-shot examples (0 for zero-shot)
        legal_prompt: Legal definition prompt
        definitions: Definition prompts
        max_seq_len: Maximum sequence length
        output_dir: Directory to save results
    
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
    elif train_data is not None:
        train_list = list(train_data)
    else:
        train_list = []
    
    results = []
    
    # Determine options based on dataset
    if dataset == 'gqa':
        options = test_list[0].get('options', ['A', 'B', 'C', 'D', 'E'])
    else:
        options = ['No', 'Yes', 'Unsure']
    
    for item in tqdm(test_list, desc=f"Prompting ({shots}-shot)"):
        # Sample few-shot examples if needed
        if shots > 0 and train_list:
            few_shot_examples = random.sample(train_list, min(shots, len(train_list)))
        else:
            few_shot_examples = None
        
        # Format prompt
        prompt = format_prompt(
            item, dataset,
            few_shot_examples=few_shot_examples,
            legal_prompt=legal_prompt,
            definitions=definitions
        )
        
        # Truncate if needed
        tokenized = model.to_tokens(prompt)
        if tokenized.shape[1] > max_seq_len:
            tokenized = tokenized[:, -max_seq_len:]
        
        # Forward pass
        logits = model.forward(
            tokenized,
            return_type="logits",
            prepend_bos=True,
            padding_side='left',
        ).detach().cpu()
        
        # Decode
        decoded = clip_decode(logits[:, -1], model.tokenizer)
        probs = prob_map(logits, options, model.tokenizer)
        
        results.append({
            'decoded': decoded,
            'probs': probs,
            'logits': logits,
            'gold': item.get('gold_distribution', item.get('label', item.get('legal_majority')))
        })
    
    model.reset_hooks()
    
    # Compute metrics
    metrics = compute_prompting_metrics(results, dataset, model.tokenizer)
    
    # Save results if output_dir provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save predictions
        preds_df = pd.DataFrame({
            'prediction': [model.tokenizer.decode(r['decoded'][1][0, 0].item()).strip() 
                          for r in results],
            'probs': [r['probs'] for r in results],
            'gold': [r['gold'] for r in results]
        })
        preds_df.to_csv(os.path.join(output_dir, f'prompting_{shots}shot_predictions.csv'), index=False)
        
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(os.path.join(output_dir, f'prompting_{shots}shot_metrics.csv'), index=False)
        
        logger.info(f"Saved results to {output_dir}")
    
    return results, pd.DataFrame(metrics)


def compute_prompting_metrics(
    results: List[Dict],
    dataset: str,
    tokenizer: AutoTokenizer
) -> Dict[str, List]:
    """
    Compute evaluation metrics for prompting results.
    
    Args:
        results: List of result dictionaries
        dataset: Dataset name
        tokenizer: Tokenizer for decoding
    
    Returns:
        Dictionary of metrics
    """
    mapping = LABEL_MAPPINGS.get(dataset, {})
    
    metrics = {
        'setting': ['prompting'],
        'macro-f1': [],
        'micro-f1': [],
        'weighted-f1': [],
    }
    
    if dataset != 'gqa':
        metrics['precision'] = []
        metrics['recall'] = []
    
    # Extract predictions and gold labels
    predictions = []
    gold_labels = []
    
    for r in results:
        # Get predicted token
        pred_token = tokenizer.decode(r['decoded'][1][0, 0].item()).strip().lower()
        
        if dataset == 'gqa':
            pred = LABEL_MAPPINGS.get('gqa', {}).get(pred_token.upper(), -1)
            gold = np.argmax(r['gold']) if isinstance(r['gold'], (list, np.ndarray)) else r['gold']
        else:
            pred = mapping.get(pred_token, -1)
            if pred == -1:
                # Try numeric
                pred = mapping.get(pred_token.strip(), 0)
            gold = r['gold'] if isinstance(r['gold'], int) else mapping.get(str(r['gold']).lower(), 0)
        
        predictions.append(0 if pred < 0 else pred)
        gold_labels.append(gold)
    
    # Compute F1 scores
    metrics['macro-f1'].append(f1_score(gold_labels, predictions, average='macro', zero_division=0))
    metrics['micro-f1'].append(f1_score(gold_labels, predictions, average='micro', zero_division=0))
    metrics['weighted-f1'].append(f1_score(gold_labels, predictions, average='weighted', zero_division=0))
    
    if dataset != 'gqa':
        metrics['precision'].append(precision_score(gold_labels, predictions, average='macro', zero_division=0))
        metrics['recall'].append(recall_score(gold_labels, predictions, average='macro', zero_division=0))
    
    logger.info(f"Prompting metrics: {metrics}")
    return metrics


def load_legal_prompt(
    def_file: str,
    legal_term: str = 'Defamation'
) -> str:
    """
    Load legal definition prompt for misinformation dataset.
    
    Args:
        def_file: Path to definitions CSV
        legal_term: Legal term to use
    
    Returns:
        Formatted legal prompt string
    """
    if not def_file or not os.path.exists(def_file):
        return ''
    
    defs = pd.read_csv(def_file)
    
    if legal_term and legal_term in defs['term'].values:
        d = defs[defs['term'] == legal_term].iloc[0].to_dict()
        defenses = d.get('defenses', 'N/A')
        if pd.isna(defenses):
            defenses = 'N/A'
        
        return f"From a legal perspective, misinformation can be problematic due to {d['term']}, defined as:\n{d['text']}{defenses}"
    
    return ''
