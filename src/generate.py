"""
Generation and steering vector computation for SAE-PD.
Contains functions for steering vector extraction and steered generation.
"""

import os
import json
import logging
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import numpy as np
from datasets import Dataset
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from sae_lens import SAE

from utils import (
    clip_decode, prob_map, dvd_decoding, create_dataset, trim_by_tokens,
    IND_TO_OPTION_MAP, INSTRUCTION_TEMPLATES
)

logger = logging.getLogger(__name__)

# Model-specific colon tokens for position detection
# These are the token IDs for ":" in different tokenizers
COLON_TOKENS = {
    'meta-llama/Llama-3.1-8B': [25],
    'google/gemma-2-9b': [235292],
    'google/gemma-2-9b-it': [199],
}


def get_colon_tokens(model_name: str) -> List[int]:
    """Get the colon token IDs for a specific model."""
    for key in COLON_TOKENS:
        if key in model_name:
            return COLON_TOKENS[key]
    # Default to Llama colon token
    return [25]


# =============================================================================
# Steering Vector Computation
# =============================================================================

def get_steering_single(
    model: HookedTransformer,
    dataset: Dataset,
    sae: SAE,
    pos_col: str = 'pos',
    neu_col: str = 'anchor',
    b: int = 4,
    num_steer: int = 200,
    colon_tokens: List[int] = None,
    model_name: str = None
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """
    Compute a single steering vector from positive/neutral pairs.
    
    Args:
        model: HookedTransformer model
        dataset: Dataset with pos_col and neu_col columns
        sae: Sparse Autoencoder
        pos_col: Column name for positive examples
        neu_col: Column name for neutral/anchor examples
        b: Batch size
        num_steer: Number of samples to use for steering
        colon_tokens: Token IDs to use as position markers (auto-detected if None)
        model_name: Model name for auto-detecting colon tokens
    
    Returns:
        Tuple of (l0_activations, steering_vector, all_vectors)
    """
    if colon_tokens is None:
        # Auto-detect colon tokens based on model name
        if model_name:
            colon_tokens = get_colon_tokens(model_name)
        else:
            colon_tokens = [25]  # Default colon token for Llama
    
    pos_tokens = create_dataset(dataset, pos_col, model.tokenizer, max_len=250)
    neu_tokens = create_dataset(dataset, neu_col, model.tokenizer, max_len=250)
    
    sae.eval()
    l0 = None
    vectors = []
    
    num_batches = min(num_steer // b, len(dataset) // b)
    
    with torch.no_grad():
        for i in range(num_batches):
            batch_pos = pos_tokens[b*i:b*i+b]["tokens"]
            batch_neg = neu_tokens[b*i:b*i+b]["tokens"]
            # Stack the list of tensors into a batch, then concatenate pos and neg
            batch_pos = torch.stack(batch_pos) if isinstance(batch_pos, list) else batch_pos
            batch_neg = torch.stack(batch_neg) if isinstance(batch_neg, list) else batch_neg
            batch_tokens = torch.concat([batch_pos, batch_neg])
            
            _, cache = model.run_with_cache(batch_tokens, prepend_bos=True)
            
            # Encode with SAE - handle both old and new sae_lens API
            hook_name = getattr(sae.cfg, 'hook_name', None) or getattr(sae.cfg.metadata, 'hook_name', None)
            feature_acts = sae.encode(cache[hook_name])
            
            del cache
            
            # Extract token representations at colon position
            reps = []
            for j in range(feature_acts.shape[0]):
                # Find last colon token position
                indices = (deepcopy(batch_tokens[j]).apply_(
                    lambda x: x in colon_tokens
                ).bool().nonzero(as_tuple=True))
                if len(indices[0]) > 0:
                    ind = indices[0][-1].item()
                else:
                    ind = -1
                reps.append(feature_acts[j, min(ind + 1, feature_acts.shape[1] - 1)])
            
            rep = torch.stack(reps)
            
            # Compute difference vectors - the batch is structured [pos, neg] so this difference is equivalent to pos - neg
            diff = rep[:b,] - rep[b:,]
            vectors.append(diff)
            
            # Track L0 sparsity
            if l0 is not None:
                l0 = torch.concat(((rep > 0).float().sum(-1).detach(), l0))
            else:
                l0 = (rep > 0).float().sum(-1).detach()
            
            del feature_acts
    
    logger.info(f"Average L0: {l0.mean().item():.2f}")
    
    steering = torch.concat(vectors).mean(dim=0)
    
    return l0, steering, vectors


def get_steering_multi(
    model: HookedTransformer,
    dataset: Dataset,
    sae: SAE,
    d: str,
    pos_col: str = 'pos',
    neu_col: str = 'anchor',
    b: int = 4,
    num_ans: int = 1,
    num_steer: int = 200,
    num_comments: int = 6,
    doc_type: Optional[str] = None,
    comment_pool: Optional[Dict] = None,
    model_name: str = None
) -> List[torch.Tensor]:
    """
    Compute multiple steering vectors from different comment sources.
    
    Args:
        model: HookedTransformer model
        dataset: Dataset with text data
        sae: Sparse Autoencoder
        d: Dataset name ('gqa', 'mislc', 'hatespeech')
        pos_col: Column name for positive examples
        neu_col: Column name for neutral examples
        b: Batch size
        num_ans: Number of answers to sample
        num_steer: Number of samples per steering vector
        num_comments: Number of comment sources
        doc_type: Document type for formatting
        comment_pool: Pre-loaded comment pool
        model_name: Model name for auto-detecting colon tokens
    
    Returns:
        List of steering vectors
    """
    steering_vectors = []
    
    for comment_idx in range(num_comments):
        # Format dataset with specific comment
        def form_with_comment(item):
            comments = item.get('comments', [''] * num_comments)
            comment = comments[comment_idx] if comment_idx < len(comments) else 'N/A'
            
            if d == 'gqa':
                pos_text = f"You are from the country of {item.get('attribute', '')}, respond to the following instruction with the help of a passage.\n\nPassage: {comment}\n\n{item.get('text', item.get('question', ''))} {item.get('rand_ans', [''])[0]}"
                anchor_text = f"You are from the country of {item.get('attribute', '')}, respond to the following instruction.\n\n{item.get('text', item.get('question', ''))} "
            elif d == 'mislc':
                pos_text = f"Claim: {item.get('claim', '')}\n\nDoes this claim contain misinformation? Answer Yes, No, or Unsure.\n\nThinking: {comment}\nAnswer: "
                anchor_text = f"Claim: {item.get('claim', '')}\n\nDoes this claim contain misinformation? Answer Yes, No, or Unsure.\nAnswer: "
            elif d == 'hatespeech':
                pos_text = f"Post: {item.get('text', '')}\n\nDoes this post violate the above hate speech policy? Answer Yes, No, or Unsure.\n\nThinking: {comment}\nAnswer: "
                anchor_text = f"Post: {item.get('text', '')}\n\nDoes this post violate the above hate speech policy? Answer Yes, No, or Unsure.\nAnswer: "
            else:
                pos_text = item.get(pos_col, '')
                anchor_text = item.get(neu_col, '')
            
            return {pos_col: pos_text, neu_col: anchor_text}
        
        formatted_dataset = dataset.map(form_with_comment)
        
        _, steering, _ = get_steering_single(
            model, formatted_dataset, sae,
            pos_col=pos_col, neu_col=neu_col,
            b=b, num_steer=num_steer,
            model_name=model_name
        )
        steering_vectors.append(steering)
        
        logger.info(f"Computed steering vector {comment_idx + 1}/{num_comments}")
    
    return steering_vectors


# =============================================================================
# Steered Generation
# =============================================================================

def create_steering_hook(
    sae: SAE,
    steering_vector: torch.Tensor,
    coeff: float = 1.0,
    position_start: int = 50
):
    """
    Create a steering hook function for model forward pass.
    
    Args:
        sae: Sparse Autoencoder
        steering_vector: Steering vector to apply
        coeff: Steering coefficient
        position_start: Position from which to apply steering
    
    Returns:
        Hook function
    """
    def steering_hook(resid_pre, hook):
        if resid_pre.shape[1] == 1:
            return
        
        feature_act = sae.encode(resid_pre)
        feature_act[:, position_start:] += coeff * steering_vector
        resid_pre = sae.decode(feature_act)
        
        return resid_pre
    
    return steering_hook


def steer_gen(
    test_data: Union[Dataset, List[dict]],
    model: HookedTransformer,
    layer: int,
    d: str,
    decoding: str = 'vanilla',
    avg_strat: str = 'logit',
    base_logits: Optional[Dict] = None,
    coeff: float = 1.0,
    sae: Optional[SAE] = None,
    steering_vecs: Optional[List[torch.Tensor]] = None,
    cache_file: Optional[str] = None,
    alpha: float = 0.2,
    tau: float = 0.4
) -> Tuple[List, Dict]:
    """
    Generate with optional steering and contrastive decoding.
    
    Args:
        test_data: Test dataset or list
        model: HookedTransformer model
        layer: Layer to apply steering (0 for no steering)
        d: Dataset name
        decoding: Decoding strategy ('vanilla', 'cad', 'dvd')
        avg_strat: Averaging strategy ('logit', 'vector')
        base_logits: Pre-computed base logits for contrastive decoding
        coeff: Steering coefficient
        sae: Sparse Autoencoder (required if layer > 0)
        steering_vecs: List of steering vectors
        cache_file: Path to cache results
        alpha: Contrastive decoding alpha
        tau: Temperature for DVD
    
    Returns:
        Tuple of (results, base_logits_dict)
    """
    # Check for cached results
    if cache_file and os.path.exists(cache_file):
        logger.info(f"Loading cached results from {cache_file}")
        with open(cache_file, 'r') as f:
            return json.load(f), {}
    
    # Convert to list if needed
    if isinstance(test_data, Dataset):
        test_list = [test_data[i] for i in range(len(test_data))]
    else:
        test_list = test_data
    
    results = []
    base_logits_out = {}
    
    # Set up hooks if steering
    editing_hooks = []
    if layer > 0 and sae is not None and steering_vecs is not None:
        # Use first steering vector or average
        if avg_strat == 'vector' and len(steering_vecs) > 1:
            combined_steering = torch.stack(steering_vecs).mean(dim=0)
        else:
            combined_steering = steering_vecs[0] if len(steering_vecs) == 1 else steering_vecs[0]
        
        hook_fn = create_steering_hook(sae, combined_steering, coeff)
        editing_hooks.append((f"blocks.{layer}.hook_resid_post", hook_fn))
    
    # Format prompts based on dataset
    for i, item in enumerate(tqdm(test_list, desc="Generating")):
        if d == 'gqa':
            prompt = f"You are from the country of {item.get('attribute', '')}, respond to the following instruction.\n\n{item.get('question', item.get('text', ''))} "
            options = item.get('options', ['A', 'B', 'C', 'D', 'E'])
        elif d == 'mislc':
            prompt = f"Claim: {item.get('claim', '')}\n\nDoes this claim contain misinformation? Answer Yes, No, or Unsure.\nAnswer: "
            options = ['No', 'Yes', 'Unsure']
        elif d == 'hatespeech':
            prompt = f"Post: {item.get('text', '')}\n\nDoes this post violate the above hate speech policy? Answer Yes, No, or Unsure.\nAnswer: "
            options = ['No', 'Yes', 'Unsure']
        else:
            prompt = item.get('formatted', item.get('anchor', ''))
            options = item.get('options', ['A', 'B', 'C', 'D', 'E'])
        
        tokenized = model.to_tokens(prompt)
        
        with model.hooks(fwd_hooks=editing_hooks):
            logits = model.forward(
                tokenized,
                return_type="logits",
                prepend_bos=True,
                padding_side='left',
            ).detach().cpu()
        
        # Apply decoding strategy
        if decoding == 'dvd' and steering_vecs is not None and len(steering_vecs) > 1:
            # Generate logits for each steering vector
            cond_logits = []
            for sv in steering_vecs:
                hook_fn = create_steering_hook(sae, sv, coeff)
                with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)]):
                    cond = model.forward(
                        tokenized,
                        return_type="logits",
                        prepend_bos=True,
                        padding_side='left',
                    ).detach().cpu()
                cond_logits.append(cond[0, -1, :])
            
            # Get base logits
            base = base_logits.get(i, logits[0, -1, :]) if base_logits else logits[0, -1, :]
            
            final_logits = dvd_decoding(cond_logits, base, tau=tau, alpha=alpha)
            decoded = clip_decode(final_logits, model.tokenizer)
            probs = prob_map(final_logits, options, model.tokenizer)
        
        elif decoding == 'dvd' and base_logits:
            # Contrastive decoding
            base = base_logits.get(i, logits[0, -1, :])
            final_logits = (1 + alpha) * logits[0, -1, :] - alpha * base
            decoded = clip_decode(final_logits.unsqueeze(0), model.tokenizer)
            probs = prob_map(final_logits.unsqueeze(0), options, model.tokenizer)
        
        elif avg_strat == 'logit' and steering_vecs is not None and len(steering_vecs) > 1:
            # Average probability distributions across all steering vectors (matches yeah.py behavior)
            # Run inference for each steering vector and average the probabilities
            all_probs = []
            for sv in steering_vecs:
                hook_fn = create_steering_hook(sae, sv, coeff)
                with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)]):
                    sv_logits = model.forward(
                        tokenized,
                        return_type="logits",
                        prepend_bos=True,
                        padding_side='left',
                    ).detach().cpu()
                sv_probs = prob_map(sv_logits, options, model.tokenizer)
                all_probs.append(sv_probs)
            
            # Average across all domains (np.mean like in yeah.py: np.mean(list(temp.values()), axis=0))
            probs = list(np.mean(all_probs, axis=0))
            decoded = clip_decode(logits[:, -1], model.tokenizer)  # Use last logits for decoded
            base_logits_out[i] = logits[0, -1, :]
        
        else:  # vanilla
            decoded = clip_decode(logits[:, -1], model.tokenizer)
            probs = prob_map(logits, options, model.tokenizer)
            base_logits_out[i] = logits[0, -1, :]
        
        results.append([decoded, probs, logits])
    
    model.reset_hooks()
    
    # Cache results if requested
    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        # Can't cache tensors directly, so save predictions
        cache_data = [[r[1]] for r in results]  # Save probability distributions
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
    
    return results, base_logits_out


def generate_with_steering(
    model: HookedTransformer,
    prompt: str,
    sae: Optional[SAE] = None,
    steering_vector: Optional[torch.Tensor] = None,
    layer: int = 0,
    coeff: float = 1.0,
    max_new_tokens: int = 50,
    temperature: float = 0.4
) -> str:
    """
    Generate text with optional steering.
    
    Args:
        model: HookedTransformer model
        prompt: Input prompt
        sae: Sparse Autoencoder
        steering_vector: Steering vector to apply
        layer: Layer to apply steering
        coeff: Steering coefficient
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Generated text
    """
    tokenized = model.to_tokens(prompt)
    
    editing_hooks = []
    if layer > 0 and sae is not None and steering_vector is not None:
        hook_fn = create_steering_hook(sae, steering_vector, coeff)
        editing_hooks.append((f"blocks.{layer}.hook_resid_post", hook_fn))
    
    with model.hooks(fwd_hooks=editing_hooks):
        output = model.generate(
            input=tokenized,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            stop_at_eos=True,
        )
    
    model.reset_hooks()
    
    return model.to_string(output[:, tokenized.shape[1]:])
