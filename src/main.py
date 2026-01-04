#!/usr/bin/env python
"""
Unified entry point for SAE-PD experiments.

This is the main interface for running all experiments across different settings,
models, and datasets.

Usage:
    python main.py --setting prompting --dataset gqa --model_name meta-llama/Llama-3.1-8B
    python main.py --setting steering --dataset mislc --layers 25 28 31 --sae_scale 8
    python main.py --setting cd --dataset hatespeech --num_comments 3
    python main.py --setting steering_cd --dataset gqa --layers 31 --decoding dvd
"""

import os
import argparse
import random
import logging
import json
from typing import Optional

from dotenv import load_dotenv
import torch
import numpy as np
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Import modules
from utils import (
    load_model_and_tokenizer, load_gqa, load_mislc, load_hs,
    load_community_comments, SAE_NAMES
)
from prompting import run_prompting, load_legal_prompt
from steering import run_steering, run_steering_sweep
from cd import run_dvd
from steering_cd import run_steering_cd, run_steering_cd_sweep

# Configure logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Environment setup - HF_TOKEN should be set in .env file
if not os.environ.get('HF_TOKEN'):
    logger.warning("HF_TOKEN not found in environment variables. Set it in .env file.")


def get_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='SAE-PD: Unified experiment runner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ==========================================================================
    # Core Settings
    # ==========================================================================
    parser.add_argument(
        '--setting', type=str, required=True,
        choices=['prompting', 'steering', 'cd', 'steering_cd'],
        help='Experiment setting to run'
    )
    parser.add_argument(
        '--model_name', type=str, default='meta-llama/Llama-3.1-8B',
        choices=['meta-llama/Llama-3.1-8B', 'google/gemma-2-9b', 'google/gemma-2-9b-it'],
        help='HuggingFace model path'
    )
    parser.add_argument(
        '--model_family', type=str, default='llama3',
        choices=['llama3', 'gemma2'],
        help='Model family (for convenience)'
    )
    parser.add_argument(
        '--dataset', type=str, required=True,
        choices=['gqa', 'mislc', 'hatespeech'],
        help='Dataset to use (gqa=GlobalOpinionQA, mislc=misinformation, hatespeech=hate speech)'
    )
    
    # ==========================================================================
    # Data Settings
    # ==========================================================================
    parser.add_argument(
        '--input_file', type=str, default='',
        help='Input data file path (relative to input directory)'
    )
    parser.add_argument(
        '--def_file', type=str, default='',
        help='Definitions file path for mislc/hatespeech'
    )
    parser.add_argument(
        '--doc_type', type=str, default='feedback',
        choices=['definition', 'feedback', 'both', 'none'],
        help='Document type for formatting'
    )
    parser.add_argument(
        '--community_setting', type=str, default='perspective',
        choices=['perspective', 'culture', 'mixed', 'w_asia', 'w_africa'],
        help='Community setting for comment loading'
    )
    
    # ==========================================================================
    # Prompting Settings
    # ==========================================================================
    parser.add_argument(
        '--fewshot', action='store_true',
        help='Enable few-shot prompting'
    )
    parser.add_argument(
        '--shots', type=int, default=3,
        help='Number of few-shot examples'
    )
    
    # ==========================================================================
    # Steering Settings
    # ==========================================================================
    parser.add_argument(
        '-l', '--layers', nargs='+', type=int, default=[31],
        help='Layers to apply steering to'
    )
    parser.add_argument(
        '--sae_scale', type=int, default=8,
        choices=[8, 32],
        help='SAE scaling factor'
    )
    parser.add_argument(
        '--coeff', type=float, default=1.0,
        help='Steering coefficient'
    )
    parser.add_argument(
        '--num_steer', type=int, default=200,
        help='Number of samples for steering vector computation'
    )
    parser.add_argument(
        '--num_comments', type=int, default=-1,
        help='Number of comment sources (-1=all, 0=combine all)'
    )
    parser.add_argument(
        '--num_ans', type=int, default=1,
        help='Number of answer samples per question'
    )
    parser.add_argument(
        '--avg_strat', type=str, default='logit',
        choices=['logit', 'vector', 'all'],
        help='Averaging strategy for multiple steering vectors'
    )
    parser.add_argument(
        '--granularity', type=str, default='granular',
        choices=['granular', 'non-granular'],
        help='Granularity setting for steering'
    )
    
    # ==========================================================================
    # Sweep Settings
    # ==========================================================================
    parser.add_argument(
        '--coefficients', nargs='+', type=float, default=[0.5, 1.0, 2.0],
        help='Coefficients to sweep over (for --sweep mode)'
    )
    parser.add_argument(
        '--sae_scales', nargs='+', type=int, default=[8, 32],
        help='SAE scales to sweep over (for --sweep mode)'
    )
    parser.add_argument(
        '--num_steers', nargs='+', type=int, default=[10, 50, 200],
        help='Number of steering samples to sweep over (for --sweep mode, corresponds to feedb in original code)'
    )
    
    # ==========================================================================
    # Contrastive Decoding Settings
    # ==========================================================================
    parser.add_argument(
        '--decoding', type=str, default='vanilla',
        choices=['vanilla', 'cad', 'dvd', 'mean', 'all'],
        help='Decoding strategy'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.2,
        help='Contrastive weight for CD/DVD'
    )
    parser.add_argument(
        '--tau', type=float, default=0.4,
        help='Temperature for DVD'
    )
    parser.add_argument(
        '--topk', type=int, default=20,
        help='Top-k for entropy calculation in DVD'
    )
    
    # ==========================================================================
    # Output Settings
    # ==========================================================================
    parser.add_argument(
        '--output_dir', type=str, default='output',
        help='Output directory for results'
    )
    parser.add_argument(
        '--cache', action='store_true',
        help='Enable caching of intermediate results'
    )
    
    # ==========================================================================
    # Other Settings
    # ==========================================================================
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug mode'
    )
    parser.add_argument(
        '--sweep', action='store_true',
        help='Run hyperparameter sweep'
    )
    
    args = parser.parse_args()
    
    # Map model_family to model_name if not explicitly set
    if args.model_family == 'gemma2' and args.model_name == 'meta-llama/Llama-3.1-8B':
        args.model_name = 'google/gemma-2-9b'
    
    # Set default input files based on dataset
    if not args.input_file:
        if args.dataset == 'gqa':
            args.input_file = 'distributional_test_globalopinionqa_small'
        elif args.dataset == 'mislc':
            args.input_file = 'train.csv'
        elif args.dataset == 'hatespeech':
            args.input_file = 'legal_train.csv'
    
    return args


def setup_output_dir(args) -> str:
    """
    Create and return the output directory path.
    
    Follows a unified naming convention:
    {output_dir}/{dataset}/{model_name}/{setting}/{parameters}
    """
    cache_dir = os.environ.get('CACHE_DIR', '')
    
    # Build output path
    model_name_clean = args.model_name.replace('/', '_')
    
    out_dir = os.path.join(
        cache_dir,
        args.output_dir,
        args.dataset,
        model_name_clean,
        args.setting
    )
    
    # Add setting-specific subdirectories
    if args.setting == 'prompting':
        shots_str = f"{args.shots}shot" if args.fewshot else "0shot"
        out_dir = os.path.join(out_dir, shots_str)
    
    elif args.setting == 'steering':
        layers_str = '_'.join(map(str, args.layers))
        out_dir = os.path.join(
            out_dir,
            f"scale{args.sae_scale}_l{layers_str}_c{args.coeff}"
        )
    
    elif args.setting == 'cd':
        out_dir = os.path.join(
            out_dir,
            f"nc{args.num_comments}_a{args.alpha}_t{args.tau}"
        )
    
    elif args.setting == 'steering_cd':
        layers_str = '_'.join(map(str, args.layers))
        out_dir = os.path.join(
            out_dir,
            f"scale{args.sae_scale}_l{layers_str}_{args.decoding}_c{args.coeff}_a{args.alpha}"
        )
    
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")
    
    return out_dir


def load_data(args):
    """Load dataset based on args."""
    if args.dataset == 'gqa':
        # Determine input path
        if args.input_file.endswith('.json'):
            input_file = args.input_file
        else:
            input_file = args.input_file
        
        train, test = load_gqa(
            input_file,
            num_ans=args.num_ans,
            base_path='modular_pluralism/input/'
        )
        
        # Load community comments if available
        comment_pool = load_community_comments(
            args.input_file.replace('.json', ''),
            community_setting=args.community_setting
        )
        
        return train, test, comment_pool
    
    elif args.dataset == 'mislc':
        train, test = load_mislc(
            args.def_file,
            num_ans=args.num_ans,
            doc_type=args.doc_type
        )
        
        # Load legal prompt
        legal_prompt = load_legal_prompt(args.def_file) if args.def_file else ''
        
        return train, test, {'legal_prompt': legal_prompt}
    
    elif args.dataset == 'hatespeech':
        train, test = load_hs(
            args.def_file,
            num_ans=args.num_ans,
            doc_type=args.doc_type
        )
        
        return train, test, {}


def main():
    """Main entry point."""
    args = get_arguments()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Arguments: {args}")
    
    # Setup output directory
    out_dir = setup_output_dir(args)
    
    # Save arguments
    with open(os.path.join(out_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load model and tokenizer
    logger.info(f"Loading model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    
    # Load data
    logger.info(f"Loading dataset: {args.dataset}")
    train_data, test_data, extra_data = load_data(args)
    
    # ==========================================================================
    # Run the appropriate setting
    # ==========================================================================
    
    if args.setting == 'prompting':
        logger.info("Running prompting evaluation")
        
        legal_prompt = extra_data.get('legal_prompt', '')
        definitions = None
        if args.dataset == 'hatespeech' and hasattr(train_data, 'column_names') and 'definitions' in train_data.column_names:
            definitions = train_data[0].get('definitions', [])
        
        results, metrics = run_prompting(
            model=model,
            test_data=test_data,
            train_data=train_data if args.fewshot else None,
            dataset=args.dataset,
            shots=args.shots if args.fewshot else 0,
            legal_prompt=legal_prompt,
            definitions=definitions,
            output_dir=out_dir
        )
        
        logger.info(f"Prompting results:\n{metrics}")
    
    elif args.setting == 'steering':
        logger.info("Running steering evaluation")
        
        comment_pool = extra_data if isinstance(extra_data, dict) and 'legal_prompt' not in extra_data else None
        
        cache_dir = os.path.join(out_dir, 'cache') if args.cache else None
        
        if args.sweep:
            metrics = run_steering_sweep(
                model=model,
                test_data=test_data,
                train_data=train_data,
                dataset=args.dataset,
                layers=args.layers,
                coefficients=args.coefficients,
                sae_scales=args.sae_scales,
                num_steers=args.num_steers,
                output_dir=out_dir,
                model_name=args.model_name,
                num_comments=args.num_comments,
                doc_type=args.doc_type
            )
        else:
            results, metrics = run_steering(
                model=model,
                test_data=test_data,
                train_data=train_data,
                dataset=args.dataset,
                layers=args.layers,
                sae_scale=args.sae_scale,
                coeff=args.coeff,
                num_steer=args.num_steer,
                num_comments=args.num_comments,
                doc_type=args.doc_type,
                avg_strat=args.avg_strat,
                cache_dir=cache_dir,
                output_dir=out_dir,
                comment_pool=comment_pool,
                model_name=args.model_name
            )
        
        logger.info(f"Steering results:\n{metrics}")
    
    elif args.setting == 'cd':
        logger.info("Running contrastive decoding (DVD) evaluation")
        
        results, metrics = run_dvd(
            model=model,
            test_data=test_data,
            train_data=train_data,
            dataset=args.dataset,
            num_comments=args.num_comments if args.num_comments > 0 else 3,
            tau=args.tau,
            alpha=args.alpha,
            topk=args.topk,
            output_dir=out_dir
        )
        
        logger.info(f"DVD results:\n{metrics}")
    
    elif args.setting == 'steering_cd':
        logger.info("Running steering + contrastive decoding evaluation")
        
        if args.sweep:
            metrics = run_steering_cd_sweep(
                model=model,
                test_data=test_data,
                train_data=train_data,
                dataset=args.dataset,
                layers=args.layers,
                output_dir=out_dir,
                model_name=args.model_name,
                sae_scale=args.sae_scale,
                num_steer=args.num_steer,
                num_comments=args.num_comments if args.num_comments > 0 else 3
            )
        else:
            decoding_strategies = [args.decoding] if args.decoding != 'all' else ['dvd', 'cad', 'mean']
            
            for dec in decoding_strategies:
                results, metrics = run_steering_cd(
                    model=model,
                    test_data=test_data,
                    train_data=train_data,
                    dataset=args.dataset,
                    layers=args.layers,
                    sae_scale=args.sae_scale,
                    coeff=args.coeff,
                    num_steer=args.num_steer,
                    num_comments=args.num_comments if args.num_comments > 0 else 3,
                    tau=args.tau,
                    alpha=args.alpha,
                    topk=args.topk,
                    decoding=dec,
                    avg_strat=args.avg_strat,
                    output_dir=os.path.join(out_dir, dec) if args.decoding == 'all' else out_dir,
                    model_name=args.model_name
                )
                
                logger.info(f"Steering+{dec} results:\n{metrics}")
    
    logger.info(f"Experiment complete. Results saved to: {out_dir}")


if __name__ == '__main__':
    main()
