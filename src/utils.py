"""
Shared utilities for SAE-PD codebase.
Consolidates model loading, dataset loading, steering logic, decoding, and evaluation.
"""

import os
import json
import random
import logging
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Any, Union

from dotenv import load_dotenv
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from sae_lens import SAE
import scipy.spatial.distance
from sklearn.metrics import f1_score, precision_score, recall_score

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment setup - HF_TOKEN should be set in .env file
HF_TOKEN = os.environ.get('HF_TOKEN')
if not HF_TOKEN:
    logger.warning("HF_TOKEN not found in environment variables. Set it in .env file.")

# SAE release names for different models
SAE_NAMES = {
    'meta-llama/Llama-3.1-8B': "llama_scope_lxr_{0}x",
    'google/gemma-2-9b': "gemma-scope-9b-pt-res",
    'google/gemma-2-9b-it': "gemma-scope-9b-pt-res",
}

# Option mappings
IND_TO_OPTION_MAP = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
OPTION_TO_IND_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

# Dataset-specific label mappings
LABEL_MAPPINGS = {
    'gqa': {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4},
    'mislc': {'unsure': 2, 'no': 0, 'yes': 1, 'un': 2},
    'hatespeech': {'Does not violate': 0, 'Violates': 1, 'Meaning unclear': 2, 
                   'yes': 1, 'unsure': 2, 'no': 0, '1': 1, '0': 0, 'un': 2}
}

# Dataset-specific instruction templates
INSTRUCTION_TEMPLATES = {
    'gqa': "You are from the country of {attribute}, respond to the following instruction.\n\n{question} ",
    'mislc': "Claim: {claim}\n\nDoes this claim contain misinformation? Answer Yes, No, or Unsure.",
    'hatespeech': "Post: {text}\n\nDoes this post violate the above hate speech policy? Answer Yes, No, or Unsure."
}


# =============================================================================
# Model & Tokenizer Loading
# =============================================================================

def load_model_and_tokenizer(
    model_name: str,
    cache_dir: Optional[str] = None,
    device_map: str = 'balanced_low_0',
    dtype: torch.dtype = torch.bfloat16,
    add_bos_token: bool = True,
    truncation_side: str = 'left'
) -> Tuple[HookedTransformer, AutoTokenizer]:
    """
    Load HookedTransformer model and tokenizer.
    
    Args:
        model_name: HuggingFace model path
        cache_dir: Cache directory for model weights
        device_map: Device mapping strategy
        dtype: Model dtype
        add_bos_token: Whether to add BOS token
        truncation_side: Side to truncate on
    
    Returns:
        Tuple of (model, tokenizer)
    """
    if cache_dir is None:
        cache_dir = os.environ.get('CACHE_DIR')
    
    hf_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
        add_bos_token=add_bos_token,
        truncation_side=truncation_side
    )
    
    model = HookedTransformer.from_pretrained_no_processing(
        model_name,
        cache_dir=cache_dir,
        device_map=device_map,
        tokenizer=hf_tokenizer,
        dtype=dtype,
    ).eval()
    
    logger.info(f"Loaded model: {model_name}")
    return model, hf_tokenizer


# =============================================================================
# SAE Management
# =============================================================================

def load_sae(
    model_name: str,
    layer: int,
    sae_scale: int = 8,
    device: str = 'cuda'
) -> Tuple[SAE, dict, Any]:
    """
    Load SAE for a specific model and layer.
    
    Args:
        model_name: HuggingFace model name
        layer: Layer number to load SAE for
        sae_scale: SAE scaling factor (8 or 32)
        device: Device to load SAE on
    
    Returns:
        Tuple of (sae, cfg_dict, sparsity)
    """
    if model_name not in SAE_NAMES:
        raise ValueError(f"Unknown model: {model_name}. Supported: {list(SAE_NAMES.keys())}")
    
    release = SAE_NAMES[model_name].format(sae_scale)
    
    # Handle different model SAE ID formats
    if 'llama' in model_name.lower():
        sae_id = f"l{layer}r_{sae_scale}x"
    elif 'gemma' in model_name.lower():
        sae_id = f"layer_{layer}/width_16k/canonical"
    else:
        sae_id = f"l{layer}r_{sae_scale}x"
    
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
    )
    
    logger.info(f"Loaded SAE: {release} / {sae_id}")
    return sae, cfg_dict, sparsity


# =============================================================================
# Dataset Loading
# =============================================================================

def load_gqa(
    input_file: str,
    num_ans: int = 1,
    train_size: int = 200,
    base_path: str = "../modular_pluralism/input/"
) -> Tuple[Dataset, List[dict]]:
    """
    Load GlobalOpinionQA (pluralism) dataset.
    
    Args:
        input_file: JSON file name (without path)
        num_ans: Number of answer samples per question
        train_size: Number of training samples
        base_path: Base path to input files
    
    Returns:
        Tuple of (train_dataset, test_list)
    """
    file_path = os.path.join(base_path, input_file + ".json") if not input_file.endswith('.json') else os.path.join(base_path, input_file)
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Sample answers based on gold distribution
    for item in data:
        item['rand_ans'] = random.choices(
            item['options'], 
            weights=item['gold_distribution'], 
            k=num_ans
        )
    
    # Split train/test
    train = random.sample(data, min(train_size, len(data)))
    train_ids = {x['id'] for x in train}
    test = [x for x in data if x['id'] not in train_ids]
    
    train_dataset = Dataset.from_list(train)
    if 'question' in train_dataset.column_names:
        train_dataset = train_dataset.rename_columns({'question': 'text'})
    
    logger.info(f"Loaded GQA dataset: {len(train)} train, {len(test)} test")
    return train_dataset, test


def load_mislc(
    def_file: str,
    num_ans: int = 1,
    doc_type: str = 'feedback',
    train_path: str = "../../train.csv",
    test_path: str = "../../test.csv"
) -> Tuple[Dataset, Dataset]:
    """
    Load misinformation dataset.
    
    Args:
        def_file: Path to definitions file
        num_ans: Number of answer samples
        doc_type: Document type ('definition', 'feedback', 'both', 'none')
        train_path: Path to training data
        test_path: Path to test data
    
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Parse comments
    def parse_comments(x):
        return [(y.strip() if not pd.isna(y) else '') 
                for y in x.replace("'", "").replace('[', "").replace(']', '').split(',')]
    
    train_data['comments'] = train_data['comments'].apply(parse_comments)
    test_data['comments'] = test_data['comments'].apply(parse_comments) if 'comments' in test_data.columns else [[]] * len(test_data)
    
    # Load definitions if provided
    definitions = {}
    if def_file and os.path.exists(def_file):
        defs = pd.read_csv(def_file)
        for _, row in defs.iterrows():
            definitions[row['term']] = row['text']
    
    train_dataset = Dataset.from_pandas(train_data)
    test_dataset = Dataset.from_pandas(test_data)
    
    logger.info(f"Loaded MISLC dataset: {len(train_data)} train, {len(test_data)} test")
    return train_dataset, test_dataset


def load_hs(
    def_file: str,
    num_ans: int = 1,
    doc_type: str = 'feedback',
    train_path: str = "../../hatespeech/data/legal_train.csv",
    test_path: str = "../../hatespeech/data/legal_test.csv"
) -> Tuple[Dataset, Dataset]:
    """
    Load hate speech dataset.
    
    Args:
        def_file: Path to definitions file
        num_ans: Number of answer samples
        doc_type: Document type
        train_path: Path to training data
        test_path: Path to test data
    
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    from ast import literal_eval
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Load definitions
    defs = pd.read_csv(def_file) if def_file and os.path.exists(def_file) else None
    def_prompts = hatespeech_formatter(defs) if defs is not None else {}
    
    # Parse notes/comments
    def parse_notes(x):
        if pd.isna(x) or str(x).strip() == '[]':
            return []
        try:
            temp = literal_eval(x)
            out = []
            temp2 = []
            for y in temp:
                if len(y) > 1:
                    out.append(y)
                else:
                    temp2.append(y)
            if temp2:
                out.append(''.join(temp2))
            return out
        except:
            return []
    
    if 'notes' in train_data.columns:
        train_data['comments'] = train_data['notes'].apply(parse_notes)
    
    train_data['definitions'] = [list(def_prompts.values())] * len(train_data)
    test_data['definitions'] = [list(def_prompts.values())] * len(test_data)
    train_data['rand_ans'] = ['No'] * len(train_data)
    test_data['rand_ans'] = ['No'] * len(test_data)
    
    train_dataset = Dataset.from_pandas(train_data)
    test_dataset = Dataset.from_pandas(test_data)
    
    logger.info(f"Loaded HateSpeech dataset: {len(train_data)} train, {len(test_data)} test")
    return train_dataset, test_dataset


def hatespeech_formatter(df: pd.DataFrame) -> Dict[str, str]:
    """Format hate speech definitions into prompts."""
    out = {}
    if df is None:
        return out
    for i in range(len(df)):
        out[df.iloc[i].term] = f'{df.iloc[i].term} is defined as: {df.iloc[i].text}'
    return out


def load_community_comments(
    input_file: str,
    community_setting: str = "perspective",
    cache_dir: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    Load community LM generated comments.
    
    Args:
        input_file: Base input file name
        community_setting: One of 'perspective', 'culture', 'mixed', 'w_asia', 'w_africa'
        cache_dir: Cache directory path
    
    Returns:
        Dictionary mapping question IDs to list of comments
    """
    if cache_dir is None:
        cache_dir = os.environ.get('CACHE_DIR', '')
    
    domain_lists = {
        "perspective": ["mistral-news_center", "mistral-news_left", "mistral-news_right",
                       "mistral-reddit_center", "mistral-reddit_left", "mistral-reddit_right"],
        "culture": ["mistral-africa_culture", "mistral-asia_culture", "mistral-europe_culture", 
                   "mistral-northamerica_culture", "mistral-southamerica_culture"],
        "mixed": ["mistral-news_center", "mistral-news_left", "mistral-news_right",
                 "mistral-africa_culture", "mistral-asia_culture", "mistral-southamerica_culture"],
        "w_asia": ["mistral-news_center", "mistral-news_left", "mistral-news_right",
                  "mistral-asia_culture", "mistral-reddit_left", "mistral-reddit_right"],
        "w_africa": ["mistral-news_center", "mistral-news_left", "mistral-news_right",
                    "mistral-africa_culture", "mistral-reddit_left", "mistral-reddit_right"]
    }
    
    domain_list = domain_lists.get(community_setting, domain_lists["perspective"])
    
    comment_pool = {}
    for domain in domain_list:
        domain_file = os.path.join(cache_dir, "community_lm_msgs", f"{input_file}_{domain}.json")
        if os.path.exists(domain_file):
            with open(domain_file, 'r') as f:
                comments = json.load(f)
            for key in comments.keys():
                if key not in comment_pool:
                    comment_pool[key] = [comments[key]]
                else:
                    comment_pool[key].append(comments[key])
    
    logger.info(f"Loaded {len(comment_pool)} comment pools for {community_setting}")
    return comment_pool


# =============================================================================
# Dataset Utilities
# =============================================================================

def create_dataset(dataset, col_name, tokenizer, max_len=150, instruct=False):
    cols = dataset.column_names
    if not instruct:
        token_dataset = dataset.map(
            lambda text: tokenizer(text[col_name], 
                                          padding='max_length',
                                          padding_side='left',
                                          truncation=True,
                                          max_length=max_len,
                                          return_tensors='pt'
                                         ),
            batched=True,
        )
    else:
        token_dataset = dataset.map(
            lambda text: tokenizer.apply_chat_template([{"role": "user", "content": text[col_name]}], 
                                                       tokenize=True, 
                                                       add_generation_prompt=True, 
                                                       return_tensors="pt",
                                                      padding='max_length',
                                                      padding_side='left',
                                                      truncation=True,
                                                      max_length=max_len,
                                                      ),
            batched = True,
        )
    token_dataset = token_dataset.remove_columns(cols + ['attention_mask'])
    token_dataset = token_dataset.rename_columns({'input_ids': 'tokens'})
    token_dataset.set_format(type="torch", columns=["tokens"])
    return token_dataset


def trim_by_tokens(text: str, tokenizer: AutoTokenizer, length: int) -> str:
    """Trim text to a maximum number of tokens."""
    tokens = tokenizer(text)['input_ids'][1:length]
    return tokenizer.decode(tokens)


# =============================================================================
# Decoding Utilities
# =============================================================================

def clip_decode(
    logits: torch.Tensor,
    tokenizer: AutoTokenizer,
    topk: int = 20
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get top-k token probabilities and indices.
    
    Args:
        logits: Logits tensor of shape (batch, vocab) or (batch, seq, vocab)
        tokenizer: Tokenizer for decoding
        topk: Number of top tokens to return
    
    Returns:
        Tuple of (log_probs, indices) for top-k tokens
    """
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    values, indices = torch.topk(log_probs, topk, dim=-1)
    
    return values, indices


def prob_map(
    logits: torch.Tensor,
    options: List[str],
    tokenizer: AutoTokenizer,
    ind_to_option_map: Optional[Dict[int, str]] = None
) -> List[float]:
    """
    Map logits to probability distribution over options.
    
    Args:
        logits: Logits tensor
        options: List of option strings
        tokenizer: Tokenizer
        ind_to_option_map: Optional mapping from index to option letter
    
    Returns:
        Probability distribution over options
    """
    if ind_to_option_map is None:
        ind_to_option_map = IND_TO_OPTION_MAP
    
    probabilities = logits.softmax(dim=-1)
    if probabilities.dim() == 3:
        probabilities = probabilities[0, -1, :]
    elif probabilities.dim() == 2:
        probabilities = probabilities[0, :]
    
    top_probs = probabilities.topk(20)
    
    probs = {}
    for token, prob in zip(top_probs.indices, top_probs.values):
        probs[tokenizer.decode(token)] = prob.item()
    
    output_distribution = [0.0] * len(options)
    for j in range(len(options)):
        for key in probs:
            if ind_to_option_map.get(j, str(j)) == key.strip():
                output_distribution[j] += probs[key]
                break
    
    if sum(output_distribution) == 0:
        # Uniform distribution if no option found
        output_distribution = [1.0 / len(options)] * len(options)
    else:
        output_distribution = [x / sum(output_distribution) for x in output_distribution]
    
    return output_distribution


def prob_compile(
    results: List[Tuple],
    tokenizer: AutoTokenizer,
    options: List[str]
) -> List[List[float]]:
    """Compile probability maps from results."""
    return [prob_map(r[2] if len(r) > 2 else r[0], options, tokenizer) for r in results]


def dvd_decoding(
    logits: List[torch.Tensor],
    logits_uncond: torch.Tensor,
    tau: float = 0.4,
    topk: int = 10,
    alpha: float = 0.2,
    torch_type: torch.dtype = torch.float16
) -> torch.Tensor:
    """
    Distributional Value Decoding (DVD).
    
    Combines multiple conditional logits with an unconditional baseline
    using entropy-weighted averaging.
    
    Args:
        logits: List of conditional logits
        logits_uncond: Unconditional baseline logits
        tau: Temperature for softmax
        topk: Number of top tokens for entropy calculation
        alpha: Contrastive weight
        torch_type: Tensor dtype
    
    Returns:
        Combined logits
    """
    probas = torch.nn.functional.softmax(logits_uncond[None] / tau, dim=-1)
    
    logits_combined = []
    entropies = []
    
    for temp_logit in logits:
        probas_cond = torch.nn.functional.softmax(temp_logit[None] / tau, dim=-1)
        values, indices = torch.topk(probas_cond, topk, largest=True)
        V = temp_logit[indices]
        entropy = -(V.exp() * V.clip(-100, 0)).sum(dim=-1).item()
        entropies.append(entropy)
        
        logits_merged = (1 + alpha) * temp_logit - alpha * logits_uncond
        final_logit = torch.where(logits_uncond > -100, logits_merged, temp_logit)
        logits_combined.append(final_logit)
    
    # Entropy-weighted combination
    en = torch.nn.functional.softmax(torch.tensor(entropies, dtype=torch_type, device='cpu'), dim=0)
    logits_comb = torch.stack(logits_combined, dim=0)
    temp = torch.matmul(en, logits_comb.type(torch_type)).unsqueeze(dim=0)
    
    final_logit = torch.where(temp > -100, temp, logits_uncond)
    return final_logit


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(
    nst_results: List,
    st_results: List,
    gold_distribution: List,
    dataset: str,
    tokenizer: Optional[AutoTokenizer] = None
) -> Dict[str, Any]:
    """
    Evaluate steering results against gold distribution.
    
    Args:
        nst_results: Non-steered results
        st_results: Steered results
        gold_distribution: Gold label distributions
        dataset: Dataset name ('gqa', 'mislc', 'hatespeech')
        tokenizer: Tokenizer for decoding predictions
    
    Returns:
        Dictionary of evaluation metrics
    """
    mapping = LABEL_MAPPINGS.get(dataset, {})
    
    metrics = {
        'setting': [],
        'macro-f1': [],
        'micro-f1': [],
        'weighted-f1': [],
    }
    
    if dataset == 'gqa':
        metrics['jsd'] = []  # Jensen-Shannon divergence
    else:
        metrics['precision'] = []
        metrics['recall'] = []
    
    # Process results based on dataset type
    if dataset == 'gqa':
        gold_labels = [np.argmax(g) for g in gold_distribution]
        
        for setting, results in [('nst', nst_results), ('st', st_results)]:
            metrics['setting'].append(setting)
            
            # Get predictions and distributions
            if isinstance(results[0], (list, tuple)):
                pred_dist = [r[1] if len(r) > 1 else r[0] for r in results]
            else:
                pred_dist = results
            
            preds = [np.argmax(p) for p in pred_dist]
            
            metrics['macro-f1'].append(f1_score(gold_labels, preds, average='macro'))
            metrics['micro-f1'].append(f1_score(gold_labels, preds, average='micro'))
            metrics['weighted-f1'].append(f1_score(gold_labels, preds, average='weighted'))
            
            # Calculate JSD
            jsd_scores = [
                scipy.spatial.distance.jensenshannon(gold_distribution[i], pred_dist[i])
                for i in range(len(gold_distribution))
            ]
            metrics['jsd'].append(np.mean(jsd_scores))
    
    else:  # mislc or hatespeech
        for setting, results in [('nst', nst_results), ('st', st_results)]:
            metrics['setting'].append(setting)
            
            # Extract predictions
            if tokenizer and isinstance(results[0], (list, tuple)):
                preds = []
                for r in results:
                    if hasattr(r[0], 'shape'):
                        token = tokenizer.decode(r[1][0, 0].item()).strip().lower()
                    else:
                        token = str(r).strip().lower()
                    preds.append(mapping.get(token, -1))
            else:
                preds = results
            
            # Ensure we have gold labels
            if isinstance(gold_distribution[0], (list, np.ndarray)):
                gold_labels = [np.argmax(g) for g in gold_distribution]
            else:
                gold_labels = list(gold_distribution)
            
            # Replace invalid predictions with 0
            preds = [(0 if p < 0 else p) for p in preds]
            
            metrics['macro-f1'].append(f1_score(gold_labels, preds, average='macro'))
            metrics['micro-f1'].append(f1_score(gold_labels, preds, average='micro'))
            metrics['weighted-f1'].append(f1_score(gold_labels, preds, average='weighted'))
            metrics['precision'].append(precision_score(gold_labels, preds, average='macro', zero_division=0))
            metrics['recall'].append(recall_score(gold_labels, preds, average='macro', zero_division=0))
    
    logger.info(f"Evaluation results: {metrics}")
    return metrics


def save_results(
    df: pd.DataFrame,
    metrics: Dict[str, Any],
    output_dir: str,
    prefix: str = ""
) -> None:
    """Save results dataframe and metrics to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    df_path = os.path.join(output_dir, f"{prefix}results.csv")
    metrics_path = os.path.join(output_dir, f"{prefix}metrics.csv")
    
    df.to_csv(df_path, index=False)
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    
    logger.info(f"Saved results to {output_dir}")
