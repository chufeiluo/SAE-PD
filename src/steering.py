"""
Steering module for SAE-PD.
Handles SAE-based steering for different datasets.
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
    clip_decode, prob_map, create_dataset, load_sae, trim_by_tokens,
    LABEL_MAPPINGS, IND_TO_OPTION_MAP, evaluate, save_results
)
from generate import (
    get_steering_single, get_steering_multi, create_steering_hook, steer_gen
)

logger = logging.getLogger(__name__)


# =============================================================================
# Steering Evaluation
# =============================================================================

def run_steering(
    model: HookedTransformer,
    test_data: Union[Dataset, List[Dict]],
    train_data: Union[Dataset, List[Dict]],
    dataset: str,
    layers: List[int],
    sae_scale: int = 8,
    coeff: float = 1.0,
    num_steer: int = 200,
    num_comments: int = -1,
    doc_type: str = 'feedback',
    avg_strat: str = 'logit',
    cache_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    comment_pool: Optional[Dict] = None,
    model_name: str = 'meta-llama/Llama-3.1-8B'
) -> Tuple[Dict[int, List], pd.DataFrame]:
    """
    Run steering evaluation across multiple layers.
    
    Args:
        model: HookedTransformer model
        test_data: Test dataset
        train_data: Training dataset for steering vectors
        dataset: Dataset name ('gqa', 'mislc', 'hatespeech')
        layers: List of layers to evaluate
        sae_scale: SAE scaling factor (8 or 32)
        coeff: Steering coefficient
        num_steer: Number of samples for steering vector computation
        num_comments: Number of comment sources (-1 for all)
        doc_type: Document type for formatting
        avg_strat: Averaging strategy ('logit', 'vector')
        cache_dir: Cache directory for intermediate results
        output_dir: Output directory for final results
        comment_pool: Pre-loaded comment pool
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
    
    # Determine number of comments
    if num_comments == -1:
        if 'comments' in train_dataset.column_names:
            num_comments = max([len(train_dataset[i].get('comments', [])) for i in range(len(train_dataset))])
        else:
            num_comments = 1
    
    logger.info(f"Using {num_comments} comment sources for steering")
    
    # First, run baseline (no steering)
    logger.info("Running baseline (no steering)")
    base_results, base_logits = steer_gen(
        test_list, model, layer=0, d=dataset,
        decoding='vanilla'
    )
    
    all_layer_results = {}
    all_metrics = []
    
    for layer in layers:
        logger.info(f"Processing layer {layer}")
        
        # Load SAE
        try:
            sae, cfg_dict, sparsity = load_sae(model_name, layer, sae_scale)
        except Exception as e:
            logger.error(f"Failed to load SAE for layer {layer}: {e}")
            continue
        
        # Set up cache
        layer_cache_dir = None
        if cache_dir:
            layer_cache_dir = os.path.join(cache_dir, f'{sae_scale}x_l-{layer}')
            os.makedirs(layer_cache_dir, exist_ok=True)
        
        # Compute steering vectors
        if num_comments == 1:
            _, steering, vectors = get_steering_single(
                model, train_dataset, sae,
                pos_col='pos', neu_col='anchor',
                b=2, num_steer=num_steer,
                model_name=model_name
            )
            steering_vectors = [steering]
        else:
            steering_vectors = get_steering_multi(
                model, train_dataset, sae, dataset,
                pos_col='pos', neu_col='anchor',
                b=2, num_steer=num_steer,
                num_comments=num_comments,
                doc_type=doc_type,
                comment_pool=comment_pool,
                model_name=model_name
            )
        
        # Cache steering vectors if requested
        if layer_cache_dir:
            vec_file = os.path.join(layer_cache_dir, f'vecs_{num_steer}_{num_comments}.pkl')
            with open(vec_file, 'wb') as f:
                pkl.dump(steering_vectors, f)
            logger.info(f"Cached steering vectors to {vec_file}")
        
        # Run steered generation
        cache_file = None
        if layer_cache_dir:
            cache_file = os.path.join(layer_cache_dir, f'st_{num_steer}_{num_comments}_{avg_strat}_{doc_type}.json')
        
        st_results, _ = steer_gen(
            test_list, model, layer, dataset,
            decoding='vanilla', avg_strat=avg_strat,
            base_logits=base_logits,
            coeff=coeff, sae=sae,
            steering_vecs=steering_vectors,
            cache_file=cache_file
        )
        
        all_layer_results[layer] = st_results
        
        # Evaluate
        gold_dist = [item.get('gold_distribution', item.get('label', item.get('legal_majority'))) 
                    for item in test_list]
        
        try:
            metrics = evaluate(base_results, st_results, gold_dist, dataset, model.tokenizer)
            metrics['layer'] = [layer, layer]
            metrics['sae_scale'] = [sae_scale, sae_scale]
            metrics['coeff'] = [coeff, coeff]
            all_metrics.append(pd.DataFrame(metrics))
        except Exception as e:
            logger.error(f"Evaluation failed for layer {layer}: {e}")
        
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
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        metrics_df.to_csv(os.path.join(output_dir, 'steering_metrics.csv'), index=False)
        
        # Save per-layer results
        for layer, results in all_layer_results.items():
            layer_df = create_results_dataframe(results, test_list, dataset, model.tokenizer)
            layer_df.to_csv(os.path.join(output_dir, f'steering_layer_{layer}_results.csv'), index=False)
        
        logger.info(f"Saved results to {output_dir}")
    
    return all_layer_results, metrics_df


def create_results_dataframe(
    results: List,
    test_data: List[Dict],
    dataset: str,
    tokenizer: AutoTokenizer
) -> pd.DataFrame:
    """
    Create a results dataframe from steering results.
    
    Args:
        results: Steering results
        test_data: Test data items
        dataset: Dataset name
        tokenizer: Tokenizer for decoding
    
    Returns:
        Results dataframe
    """
    mapping = LABEL_MAPPINGS.get(dataset, {})
    
    data = {
        'prediction': [],
        'pred_dist': [],
    }
    
    if dataset == 'gqa':
        data['gold_dist'] = []
        data['gold_label'] = []
    else:
        data['gold_label'] = []
    
    for i, (result, item) in enumerate(zip(results, test_data)):
        # Extract prediction
        pred_token = tokenizer.decode(result[0][1][0, 0].item()).strip()
        data['prediction'].append(pred_token)
        
        # Extract probability distribution
        if len(result) > 1:
            data['pred_dist'].append(result[1])
        else:
            data['pred_dist'].append(None)
        
        # Extract gold
        if dataset == 'gqa':
            gold_dist = item.get('gold_distribution', [])
            data['gold_dist'].append(gold_dist)
            data['gold_label'].append(np.argmax(gold_dist) if gold_dist else -1)
        else:
            data['gold_label'].append(item.get('label', item.get('legal_majority', -1)))
    
    return pd.DataFrame(data)


def run_steering_sweep(
    model: HookedTransformer,
    test_data: Union[Dataset, List[Dict]],
    train_data: Union[Dataset, List[Dict]],
    dataset: str,
    layers: List[int],
    coefficients: List[float] = [0.5, 1.0, 2.0],
    sae_scales: List[int] = [8, 32],
    num_steers: List[int] = [10, 50, 200],
    output_dir: Optional[str] = None,
    model_name: str = 'meta-llama/Llama-3.1-8B',
    **kwargs
) -> pd.DataFrame:
    """
    Run a sweep over steering hyperparameters.
    
    Args:
        model: HookedTransformer model
        test_data: Test dataset
        train_data: Training dataset
        dataset: Dataset name
        layers: Layers to evaluate
        coefficients: Steering coefficients to try
        sae_scales: SAE scales to try
        num_steers: Number of steering samples to try (corresponds to feedb in yeah.py)
        output_dir: Output directory
        model_name: Model name
        **kwargs: Additional arguments for run_steering
    
    Returns:
        Combined metrics dataframe
    """
    all_metrics = []
    
    for sae_scale in sae_scales:
        for coeff in coefficients:
            for num_steer in num_steers:
                logger.info(f"Running sweep: sae_scale={sae_scale}, coeff={coeff}, num_steer={num_steer}")
                
                sweep_output = None
                if output_dir:
                    sweep_output = os.path.join(output_dir, f'scale_{sae_scale}_coeff_{coeff}_ns{num_steer}')
                
                _, metrics = run_steering(
                    model, test_data, train_data, dataset,
                    layers=layers,
                    sae_scale=sae_scale,
                    coeff=coeff,
                    num_steer=num_steer,
                    output_dir=sweep_output,
                    model_name=model_name,
                    **kwargs
                )
                
                # Add sweep parameters to metrics
                if not metrics.empty:
                    metrics['num_steer'] = num_steer
                all_metrics.append(metrics)
    
    combined = pd.concat(all_metrics, ignore_index=True)
    
    if output_dir:
        combined.to_csv(os.path.join(output_dir, 'sweep_results.csv'), index=False)
    
    return combined


def analyze_steering_results(
    metrics_df: pd.DataFrame,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze steering results and find best configurations.
    
    Args:
        metrics_df: Metrics dataframe from steering runs
        output_dir: Output directory for plots
    
    Returns:
        Analysis summary
    """
    summary = {
        'best_layer': None,
        'best_coeff': None,
        'best_sae_scale': None,
        'best_f1': 0.0,
        'improvement_over_baseline': 0.0
    }
    
    if metrics_df.empty:
        return summary
    
    # Find best steering configuration
    st_results = metrics_df[metrics_df['setting'] == 'st']
    if not st_results.empty:
        best_idx = st_results['weighted-f1'].idxmax()
        best_row = st_results.loc[best_idx]
        
        summary['best_layer'] = best_row.get('layer')
        summary['best_coeff'] = best_row.get('coeff')
        summary['best_sae_scale'] = best_row.get('sae_scale')
        summary['best_f1'] = best_row['weighted-f1']
        
        # Calculate improvement
        baseline = metrics_df[metrics_df['setting'] == 'nst']['weighted-f1'].mean()
        summary['improvement_over_baseline'] = summary['best_f1'] - baseline
    
    logger.info(f"Analysis summary: {summary}")
    
    if output_dir:
        with open(os.path.join(output_dir, 'analysis_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    return summary
