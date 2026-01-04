"""
Combined Steering and Contrastive Decoding module for SAE-PD.
Integrates SAE-based steering with DVD/CAD decoding strategies.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import pickle as pkl

import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from sae_lens import SAE
from sklearn.metrics import f1_score, precision_score, recall_score
import scipy.spatial.distance

from utils import (
    clip_decode, prob_map, dvd_decoding, load_sae,
    LABEL_MAPPINGS, IND_TO_OPTION_MAP, evaluate
)
from generate import (
    get_steering_single, get_steering_multi, create_steering_hook, steer_gen
)
from cd import dvd_decode_multi, cad_decode, format_dvd_prompt

logger = logging.getLogger(__name__)


# =============================================================================
# Combined Steering + CD
# =============================================================================

def run_steering_cd(
    model: HookedTransformer,
    test_data: Union[Dataset, List[Dict]],
    train_data: Union[Dataset, List[Dict]],
    dataset: str,
    layers: List[int],
    sae_scale: int = 8,
    coeff: float = 1.0,
    num_steer: int = 200,
    num_comments: int = 3,
    tau: float = 0.4,
    alpha: float = 0.2,
    topk: int = 20,
    decoding: str = 'dvd',
    avg_strat: str = 'logit',
    output_dir: Optional[str] = None,
    model_name: str = 'meta-llama/Llama-3.1-8B'
) -> Tuple[Dict[int, List], pd.DataFrame]:
    """
    Run combined steering and contrastive decoding evaluation.
    
    This method applies SAE-based steering to generate multiple conditional
    distributions, then combines them using DVD or averages them with
    contrastive decoding.
    
    Args:
        model: HookedTransformer model
        test_data: Test dataset
        train_data: Training dataset for steering vectors
        dataset: Dataset name ('gqa', 'mislc', 'hatespeech')
        layers: List of layers to evaluate
        sae_scale: SAE scaling factor (8 or 32)
        coeff: Steering coefficient
        num_steer: Number of samples for steering vector computation
        num_comments: Number of comment sources
        tau: Temperature for DVD
        alpha: Contrastive weight
        topk: Top-k for entropy calculation
        decoding: Decoding strategy ('dvd', 'cad', 'mean')
        avg_strat: Averaging strategy ('logit', 'vector')
        output_dir: Output directory
        model_name: Model name for SAE loading
    
    Returns:
        Tuple of (layer_results dict, metrics dataframe)
    """
    # Convert to list if needed
    if isinstance(test_data, Dataset):
        test_list = [test_data[i] for i in range(len(test_data))]
    else:
        test_list = list(test_data)
    
    if isinstance(train_data, Dataset):
        train_dataset = train_data
    else:
        train_dataset = Dataset.from_list(train_data)
    
    # Determine options
    if dataset == 'gqa':
        options = test_list[0].get('options', ['A', 'B', 'C', 'D', 'E'])
    else:
        options = ['No', 'Yes', 'Unsure']
    
    # First, compute baseline (unconditional) logits
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
    
    all_layer_results = {}
    all_metrics = []
    
    for layer in layers:
        logger.info(f"Processing layer {layer} with steering + {decoding}")
        
        # Load SAE
        try:
            sae, cfg_dict, sparsity = load_sae(model_name, layer, sae_scale)
        except Exception as e:
            logger.error(f"Failed to load SAE for layer {layer}: {e}")
            continue
        
        # Compute steering vectors
        if num_comments <= 1:
            _, steering, _ = get_steering_single(
                model, train_dataset, sae,
                pos_col='pos', neu_col='anchor',
                b=2, num_steer=num_steer
            )
            steering_vectors = [steering]
        else:
            steering_vectors = get_steering_multi(
                model, train_dataset, sae, dataset,
                pos_col='pos', neu_col='anchor',
                b=2, num_steer=num_steer,
                num_comments=num_comments
            )
        
        # Run steered generation with contrastive decoding
        layer_results = []
        
        for i, item in enumerate(tqdm(test_list, desc=f"Steering+{decoding} L{layer}")):
            prompt = format_dvd_prompt(item, dataset)
            tokenized = model.to_tokens(prompt)
            
            # Generate logits with each steering vector
            conditional_logits = []
            
            for sv_idx, sv in enumerate(steering_vectors):
                hook_fn = create_steering_hook(sae, sv, coeff)
                editing_hooks = [(f"blocks.{layer}.hook_resid_post", hook_fn)]
                
                with model.hooks(fwd_hooks=editing_hooks):
                    logits = model.forward(
                        tokenized,
                        return_type="logits",
                        prepend_bos=True,
                        padding_side='left',
                    ).detach().cpu()
                
                conditional_logits.append(logits[0, -1, :])
            
            # Apply decoding strategy
            base_logits = baseline_results[i]['logits']
            
            if decoding == 'dvd':
                final_logits = dvd_decode_multi(
                    conditional_logits, base_logits,
                    tau=tau, alpha=alpha, topk=topk
                )
            elif decoding == 'cad':
                # Average conditional logits, then apply CAD
                avg_cond = torch.stack(conditional_logits).mean(dim=0)
                final_logits = cad_decode(avg_cond, base_logits, alpha)
            elif decoding == 'mean':
                # Simple mean of conditional logits
                final_logits = torch.stack(conditional_logits).mean(dim=0)
            else:
                # Fallback to first conditional
                final_logits = conditional_logits[0]
            
            decoded = clip_decode(final_logits.unsqueeze(0), model.tokenizer)
            probs = prob_map(final_logits.unsqueeze(0), options, model.tokenizer)
            
            layer_results.append({
                'decoded': decoded,
                'probs': probs,
                'logits': final_logits,
                'conditional_logits': conditional_logits,
                'gold': item.get('gold_distribution', item.get('label', item.get('legal_majority')))
            })
        
        model.reset_hooks()
        all_layer_results[layer] = layer_results
        
        # Compute metrics
        metrics = compute_steering_cd_metrics(
            baseline_results, layer_results, test_list, dataset, model.tokenizer
        )
        metrics['layer'] = [layer, layer]
        metrics['sae_scale'] = [sae_scale, sae_scale]
        metrics['coeff'] = [coeff, coeff]
        metrics['decoding'] = [decoding, decoding]
        
        all_metrics.append(pd.DataFrame(metrics))
        
        # Clean up SAE
        del sae
        torch.cuda.empty_cache()
    
    # Combine metrics
    if all_metrics:
        metrics_df = pd.concat(all_metrics, ignore_index=True)
    else:
        metrics_df = pd.DataFrame()
    
    # Save results
    if output_dir:
        save_steering_cd_results(
            all_layer_results, metrics_df, baseline_results,
            test_list, dataset, model.tokenizer, output_dir
        )
    
    return all_layer_results, metrics_df


def compute_steering_cd_metrics(
    baseline_results: List[Dict],
    steered_results: List[Dict],
    test_data: List[Dict],
    dataset: str,
    tokenizer: AutoTokenizer
) -> Dict[str, List]:
    """
    Compute metrics for steering + CD evaluation.
    
    Args:
        baseline_results: Baseline results
        steered_results: Steered + CD results
        test_data: Test data items
        dataset: Dataset name
        tokenizer: Tokenizer
    
    Returns:
        Metrics dictionary
    """
    mapping = LABEL_MAPPINGS.get(dataset, {})
    
    metrics = {
        'setting': ['nst', 'st'],
        'macro-f1': [],
        'micro-f1': [],
        'weighted-f1': [],
    }
    
    if dataset == 'gqa':
        metrics['jsd'] = []
    else:
        metrics['precision-1'] = []
        metrics['recall-1'] = []
        metrics['f1-1'] = []
    
    for setting, results in [('nst', baseline_results), ('st', steered_results)]:
        predictions = []
        pred_dists = []
        gold_labels = []
        gold_dists = []
        
        for i, (result, item) in enumerate(zip(results, test_data)):
            pred_token = tokenizer.decode(result['decoded'][1][0, 0].item()).strip().lower()
            
            if dataset == 'gqa':
                pred = LABEL_MAPPINGS.get('gqa', {}).get(pred_token.upper(), -1)
                gold_dist = item.get('gold_distribution', [])
                gold = np.argmax(gold_dist) if gold_dist else -1
                gold_dists.append(gold_dist)
                pred_dists.append(result.get('probs', []))
            else:
                pred = mapping.get(pred_token, -1)
                if pred == -1:
                    pred = 0
                gold = item.get('label', item.get('legal_majority', 0))
                if isinstance(gold, str):
                    gold = mapping.get(gold.lower(), 0)
            
            predictions.append(0 if pred < 0 else pred)
            gold_labels.append(gold)
        
        metrics['macro-f1'].append(f1_score(gold_labels, predictions, average='macro', zero_division=0))
        metrics['micro-f1'].append(f1_score(gold_labels, predictions, average='micro', zero_division=0))
        metrics['weighted-f1'].append(f1_score(gold_labels, predictions, average='weighted', zero_division=0))
        
        if dataset == 'gqa':
            # Calculate JSD
            jsd_scores = []
            for gd, pd in zip(gold_dists, pred_dists):
                if gd and pd and len(gd) == len(pd):
                    jsd_scores.append(scipy.spatial.distance.jensenshannon(gd, pd))
            metrics['jsd'].append(np.mean(jsd_scores) if jsd_scores else 1.0)
        else:
            metrics['precision-1'].append(precision_score(gold_labels, predictions, labels=[1], average='macro', zero_division=0))
            metrics['recall-1'].append(recall_score(gold_labels, predictions, labels=[1], average='micro', zero_division=0))
            metrics['f1-1'].append(f1_score(gold_labels, predictions, labels=[1], average='macro', zero_division=0))
    
    logger.info(f"Steering+CD metrics: {metrics}")
    return metrics


def save_steering_cd_results(
    layer_results: Dict[int, List],
    metrics_df: pd.DataFrame,
    baseline_results: List[Dict],
    test_data: List[Dict],
    dataset: str,
    tokenizer: AutoTokenizer,
    output_dir: str
) -> None:
    """
    Save steering + CD results to files.
    
    Args:
        layer_results: Results by layer
        metrics_df: Metrics dataframe
        baseline_results: Baseline results
        test_data: Test data items
        dataset: Dataset name
        tokenizer: Tokenizer
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics_df.to_csv(os.path.join(output_dir, 'steering_cd_metrics.csv'), index=False)
    
    # Save per-layer results
    for layer, results in layer_results.items():
        df = pd.DataFrame({
            'baseline_pred': [tokenizer.decode(r['decoded'][1][0, 0].item()).strip() for r in baseline_results],
            'steered_pred': [tokenizer.decode(r['decoded'][1][0, 0].item()).strip() for r in results],
            'baseline_probs': [r['probs'] for r in baseline_results],
            'steered_probs': [r['probs'] for r in results],
        })
        
        if dataset == 'gqa':
            df['gold_dist'] = [item.get('gold_distribution') for item in test_data]
            df['gold_label'] = [np.argmax(item.get('gold_distribution', [0])) for item in test_data]
        else:
            df['gold_label'] = [item.get('label', item.get('legal_majority', 0)) for item in test_data]
        
        df.to_csv(os.path.join(output_dir, f'steering_cd_layer_{layer}_results.csv'), index=False)
    
    logger.info(f"Saved steering+CD results to {output_dir}")


def run_steering_cd_sweep(
    model: HookedTransformer,
    test_data: Union[Dataset, List[Dict]],
    train_data: Union[Dataset, List[Dict]],
    dataset: str,
    layers: List[int],
    coefficients: List[float] = [0.5, 1.0, 2.0],
    alphas: List[float] = [0.1, 0.2, 0.5],
    taus: List[float] = [0.4, 1.0],
    decodings: List[str] = ['dvd', 'cad', 'mean'],
    output_dir: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Run a hyperparameter sweep for steering + CD.
    
    Args:
        model: HookedTransformer model
        test_data: Test dataset
        train_data: Training dataset
        dataset: Dataset name
        layers: Layers to evaluate
        coefficients: Steering coefficients
        alphas: Contrastive weights
        taus: Temperatures
        decodings: Decoding strategies
        output_dir: Output directory
        **kwargs: Additional arguments
    
    Returns:
        Combined metrics dataframe
    """
    all_metrics = []
    
    for decoding in decodings:
        for coeff in coefficients:
            for alpha in alphas:
                for tau in taus:
                    logger.info(f"Sweep: decoding={decoding}, coeff={coeff}, alpha={alpha}, tau={tau}")
                    
                    sweep_output = None
                    if output_dir:
                        sweep_output = os.path.join(
                            output_dir, 
                            f'{decoding}_c{coeff}_a{alpha}_t{tau}'
                        )
                    
                    _, metrics = run_steering_cd(
                        model, test_data, train_data, dataset,
                        layers=layers,
                        coeff=coeff,
                        alpha=alpha,
                        tau=tau,
                        decoding=decoding,
                        output_dir=sweep_output,
                        **kwargs
                    )
                    
                    metrics['alpha'] = alpha
                    metrics['tau'] = tau
                    all_metrics.append(metrics)
    
    combined = pd.concat(all_metrics, ignore_index=True)
    
    if output_dir:
        combined.to_csv(os.path.join(output_dir, 'steering_cd_sweep.csv'), index=False)
    
    return combined
