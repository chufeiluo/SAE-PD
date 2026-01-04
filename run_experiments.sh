#!/bin/bash -l
#SBATCH --account=aip-zhu2048
#SBATCH --job-name=sae_pd_experiments
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH --array=0-8
#SBATCH --output=/home/%u/projects/aip-zhu2048/%u/SAE-PD/slurm/%x_%A_%a.out
#SBATCH --error=/home/%u/projects/aip-zhu2048/%u/SAE-PD/slurm/%x_%A_%a.err

# =============================================================================
# SAE-PD Experiment Runner with SLURM Array Support
# =============================================================================
# This script runs experiments using SLURM job arrays for parallel execution.
# Each experiment configuration runs as a separate array task.
#
# Usage:
#   # Submit all experiments as a SLURM array job:
#   sbatch --array=0-N run_experiments.sh [experiment_type]
#
#   # Generate experiment configs and submit:
#   ./run_experiments.sh --submit prompting
#   ./run_experiments.sh --submit steering
#   ./run_experiments.sh --submit sweep
#
#   # List experiments without running:
#   ./run_experiments.sh --list prompting
#
#   # Run locally (non-SLURM):
#   ./run_experiments.sh --local prompting
#
# Experiment types: prompting, steering, cd, steering_cd, sweep, gemma, all
# =============================================================================

set -e  # Exit on error

# Configuration
PYTHON=${PYTHON:-python}

# Use SLURM_SUBMIT_DIR when running as SLURM job, otherwise resolve from script location
if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
fi

SRC_DIR="$SCRIPT_DIR/src"
OUTPUT_DIR=${OUTPUT_DIR:-output}
CACHE_DIR=${CACHE_DIR:-/tmp/sae-pd-cache}

# Default model and settings
MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-3.1-8B"}
SAE_SCALE=${SAE_SCALE:-8}

# Sweep hyperparameters (can be overridden via environment)
SWEEP_LAYERS=${SWEEP_LAYERS:-"25 26 27 28 29 30 31"}
SWEEP_SAE_SCALES=${SWEEP_SAE_SCALES:-"8 32"}
SWEEP_NUM_STEERS=${SWEEP_NUM_STEERS:-"10 50 200"}
SWEEP_DATASETS=${SWEEP_DATASETS:-"gqa mislc hatespeech"}

# =============================================================================
# Helper Functions
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task ${SLURM_ARRAY_TASK_ID:-local}] $1"
}

run_experiment() {
    log "Running: $PYTHON $SRC_DIR/main.py $@"
    $PYTHON $SRC_DIR/main.py "$@"
}

# =============================================================================
# Experiment Configuration Arrays
# =============================================================================
# Each function populates the EXPERIMENTS array with experiment configurations.
# Format: "setting|dataset|extra_args"
# =============================================================================

declare -a EXPERIMENTS

build_prompting_experiments() {
    EXPERIMENTS=()
    local datasets=("gqa" "mislc" "hatespeech")
    local fewshot_options=("" "--fewshot --shots 3")
    
    for dataset in "${datasets[@]}"; do
        for fewshot in "${fewshot_options[@]}"; do
            EXPERIMENTS+=("prompting|${dataset}|${fewshot}")
        done
    done
}

build_steering_experiments() {
    EXPERIMENTS=()
    local datasets=("gqa" "mislc" "hatespeech")
    local NUM_STEER=(50 100 150)
    
    for dataset in "${datasets[@]}"; do
        for num_steer in "${NUM_STEER[@]}"; do
            local extra_args="--layers $SWEEP_LAYERS --sae_scale $SAE_SCALE --coeff 1.0 --num_steer $num_steer --cache --num_comments 3"
            # if [[ "$dataset" == "mislc" ]]; then
            #     extra_args="$extra_args --num_comments 3"
            # fi
            EXPERIMENTS+=("steering|${dataset}|${extra_args}")
        done
        # local extra_args="--layers $SWEEP_LAYERS --sae_scale $SAE_SCALE --coeff 1.0 --cache"
        # if [[ "$dataset" == "gqa" ]]; then
        #     extra_args="$extra_args --num_steer 200"
        # elif [[ "$dataset" == "mislc" ]]; then
        #     extra_args="$extra_args --num_comments 3"
        # fi
        # EXPERIMENTS+=("steering|${dataset}|${extra_args}")
    done
}

build_cd_experiments() {
    EXPERIMENTS=()
    
    # GQA
    EXPERIMENTS+=("cd|gqa|--num_comments 6 --alpha 0.2 --tau 0.4")
    # Misinformation
    EXPERIMENTS+=("cd|mislc|--num_comments 3 --alpha 0.1 --tau 0.4")
    # Hate Speech
    EXPERIMENTS+=("cd|hatespeech|--num_comments 3 --alpha 0.2 --tau 0.4")
}

build_steering_cd_experiments() {
    EXPERIMENTS=()
    
    # GQA
    EXPERIMENTS+=("steering_cd|gqa|--layers 31 --sae_scale $SAE_SCALE --coeff 1.0 --decoding dvd --alpha 0.2 --tau 0.4")
    # Misinformation
    EXPERIMENTS+=("steering_cd|mislc|--layers 29 --sae_scale 32 --coeff 1.0 --decoding cad --alpha 0.1")
    # Hate Speech
    EXPERIMENTS+=("steering_cd|hatespeech|--layers 28 31 --sae_scale $SAE_SCALE --decoding dvd --alpha 0.2 --tau 0.4")
}

build_sweep_experiments() {
    EXPERIMENTS=()
    
    # Parse sweep parameters into arrays
    read -ra layers_arr <<< "$SWEEP_LAYERS"
    read -ra scales_arr <<< "$SWEEP_SAE_SCALES"
    read -ra coeffs_arr <<< "$SWEEP_COEFFICIENTS"
    read -ra steers_arr <<< "$SWEEP_NUM_STEERS"
    read -ra datasets_arr <<< "$SWEEP_DATASETS"
    
    # Generate all combinations for steering sweep
    for dataset in "${datasets_arr[@]}"; do
        for layer in "${layers_arr[@]}"; do
            for scale in "${scales_arr[@]}"; do
                for coeff in "${coeffs_arr[@]}"; do
                    for num_steer in "${steers_arr[@]}"; do
                        local extra_args="--layers $layer --sae_scale $scale --coeff $coeff --num_steer $num_steer --cache"
                        EXPERIMENTS+=("steering|${dataset}|${extra_args}")
                    done
                done
            done
        done
    done
}

build_gemma_experiments() {
    EXPERIMENTS=()
    local gemma_model="google/gemma-2-9b"
    
    # Prompting on GQA
    EXPERIMENTS+=("prompting|gqa|--model_name $gemma_model --model_family gemma2 --fewshot")
    # Steering on Misinformation
    EXPERIMENTS+=("steering|mislc|--model_name $gemma_model --model_family gemma2 --layers 28 31")
}

build_all_experiments() {
    EXPERIMENTS=()
    
    # Temporarily store experiments
    local temp_experiments=()
    
    build_prompting_experiments
    temp_experiments+=("${EXPERIMENTS[@]}")
    
    build_steering_experiments
    temp_experiments+=("${EXPERIMENTS[@]}")
    
    build_cd_experiments
    temp_experiments+=("${EXPERIMENTS[@]}")
    
    build_steering_cd_experiments
    temp_experiments+=("${EXPERIMENTS[@]}")
    
    EXPERIMENTS=("${temp_experiments[@]}")
}

# =============================================================================
# SLURM Array Task Execution
# =============================================================================

run_array_task() {
    local experiment_type="$1"
    local task_id="${SLURM_ARRAY_TASK_ID:-0}"
    
    # Build experiments based on type
    case "$experiment_type" in
        prompting)     build_prompting_experiments ;;
        steering)      build_steering_experiments ;;
        cd)            build_cd_experiments ;;
        steering_cd)   build_steering_cd_experiments ;;
        sweep)         build_sweep_experiments ;;
        gemma)         build_gemma_experiments ;;
        all)           build_all_experiments ;;
        *)
            echo "Unknown experiment type: $experiment_type"
            exit 1
            ;;
    esac
    
    local num_experiments=${#EXPERIMENTS[@]}
    
    if [[ $task_id -ge $num_experiments ]]; then
        log "Task ID $task_id exceeds number of experiments ($num_experiments). Exiting."
        exit 0
    fi
    
    # Parse experiment configuration
    local config="${EXPERIMENTS[$task_id]}"
    IFS='|' read -r setting dataset extra_args <<< "$config"
    
    log "=== Experiment $((task_id + 1))/$num_experiments ==="
    log "Setting: $setting"
    log "Dataset: $dataset"
    log "Extra args: $extra_args"
    
    # Build command
    local cmd_args="--setting $setting --dataset $dataset --model_name $MODEL_NAME --output_dir $OUTPUT_DIR"
    
    # Add extra args (handle model_name override in extra_args)
    if [[ "$extra_args" == *"--model_name"* ]]; then
        # Remove default model_name if overridden
        cmd_args="--setting $setting --dataset $dataset --output_dir $OUTPUT_DIR"
    fi
    
    # Run the experiment
    run_experiment $cmd_args $extra_args
    
    log "Experiment complete!"
}

# =============================================================================
# Utility Functions
# =============================================================================

list_experiments() {
    local experiment_type="$1"
    
    case "$experiment_type" in
        prompting)     build_prompting_experiments ;;
        steering)      build_steering_experiments ;;
        cd)            build_cd_experiments ;;
        steering_cd)   build_steering_cd_experiments ;;
        sweep)         build_sweep_experiments ;;
        gemma)         build_gemma_experiments ;;
        all)           build_all_experiments ;;
        *)
            echo "Unknown experiment type: $experiment_type"
            exit 1
            ;;
    esac
    
    echo "=== Experiments for '$experiment_type' ==="
    echo "Total: ${#EXPERIMENTS[@]} experiments"
    echo ""
    
    for i in "${!EXPERIMENTS[@]}"; do
        IFS='|' read -r setting dataset extra_args <<< "${EXPERIMENTS[$i]}"
        echo "[$i] $setting | $dataset | $extra_args"
    done
    
    echo ""
    echo "Submit with: sbatch --array=0-$((${#EXPERIMENTS[@]} - 1)) $0 $experiment_type"
}

submit_experiments() {
    local experiment_type="$1"
    
    case "$experiment_type" in
        prompting)     build_prompting_experiments ;;
        steering)      build_steering_experiments ;;
        cd)            build_cd_experiments ;;
        steering_cd)   build_steering_cd_experiments ;;
        sweep)         build_sweep_experiments ;;
        gemma)         build_gemma_experiments ;;
        all)           build_all_experiments ;;
        *)
            echo "Unknown experiment type: $experiment_type"
            exit 1
            ;;
    esac
    
    local num_experiments=${#EXPERIMENTS[@]}
    local array_range="0-$((num_experiments - 1))"
    
    echo "Submitting $num_experiments experiments as SLURM array job..."
    echo "Array range: $array_range"
    
    sbatch --array="$array_range" "$SCRIPT_DIR/run_experiments.sh" "$experiment_type"
}

run_local() {
    local experiment_type="$1"
    
    case "$experiment_type" in
        prompting)     build_prompting_experiments ;;
        steering)      build_steering_experiments ;;
        cd)            build_cd_experiments ;;
        steering_cd)   build_steering_cd_experiments ;;
        sweep)         build_sweep_experiments ;;
        gemma)         build_gemma_experiments ;;
        all)           build_all_experiments ;;
        *)
            echo "Unknown experiment type: $experiment_type"
            exit 1
            ;;
    esac
    
    local num_experiments=${#EXPERIMENTS[@]}
    log "Running $num_experiments experiments locally (sequentially)..."
    
    for i in "${!EXPERIMENTS[@]}"; do
        IFS='|' read -r setting dataset extra_args <<< "${EXPERIMENTS[$i]}"
        
        log "=== Experiment $((i + 1))/$num_experiments ==="
        log "Setting: $setting, Dataset: $dataset"
        
        local cmd_args="--setting $setting --dataset $dataset --model_name $MODEL_NAME --output_dir $OUTPUT_DIR"
        
        if [[ "$extra_args" == *"--model_name"* ]]; then
            cmd_args="--setting $setting --dataset $dataset --output_dir $OUTPUT_DIR"
        fi
        
        run_experiment $cmd_args $extra_args
    done
    
    log "All experiments complete!"
}

# =============================================================================
# Main Entry Point
# =============================================================================

print_usage() {
    echo "Usage: $0 [options] <experiment_type>"
    echo ""
    echo "Experiment types:"
    echo "  prompting    - Zero-shot and few-shot prompting experiments"
    echo "  steering     - SAE steering experiments"
    echo "  cd           - Contrastive decoding experiments"
    echo "  steering_cd  - Steering + contrastive decoding experiments"
    echo "  sweep        - Hyperparameter sweep (generates many experiments)"
    echo "  gemma        - Gemma model experiments"
    echo "  all          - All experiments combined"
    echo ""
    echo "Options:"
    echo "  --list       List all experiments without running"
    echo "  --submit     Submit experiments as SLURM array job"
    echo "  --local      Run all experiments locally (sequentially)"
    echo "  --help       Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  MODEL_NAME         - Model to use (default: meta-llama/Llama-3.1-8B)"
    echo "  SAE_SCALE          - SAE scale (default: 8)"
    echo "  OUTPUT_DIR         - Output directory (default: output)"
    echo "  SWEEP_LAYERS       - Layers for sweep (default: 25 28 31)"
    echo "  SWEEP_SAE_SCALES   - SAE scales for sweep (default: 8 32)"
    echo "  SWEEP_COEFFICIENTS - Coefficients for sweep (default: 0.5 1.0 2.0)"
    echo "  SWEEP_NUM_STEERS   - Num steers for sweep (default: 10 50 200)"
    echo "  SWEEP_DATASETS     - Datasets for sweep (default: gqa mislc hatespeech)"
    echo ""
    echo "Examples:"
    echo "  # List all prompting experiments:"
    echo "  $0 --list prompting"
    echo ""
    echo "  # Submit steering experiments as SLURM array:"
    echo "  $0 --submit steering"
    echo ""
    echo "  # Run sweep experiments locally:"
    echo "  $0 --local sweep"
    echo ""
    echo "  # Direct SLURM array submission:"
    echo "  sbatch --array=0-5 $0 prompting"
}

main() {
    # Check if running as SLURM array task
    if [[ -n "$SLURM_ARRAY_TASK_ID" ]]; then
        # Running as SLURM array task - execute single experiment
        if [[ $# -eq 0 ]]; then
            echo "Error: experiment type required when running as SLURM array task"
            exit 1
        fi
        run_array_task "$1"
        exit 0
    fi
    
    # Parse command line arguments
    local mode="help"
    local experiment_type=""
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --list)
                mode="list"
                shift
                ;;
            --submit)
                mode="submit"
                shift
                ;;
            --local)
                mode="local"
                shift
                ;;
            --help|-h)
                mode="help"
                shift
                ;;
            prompting|steering|cd|steering_cd|sweep|gemma|all)
                experiment_type="$1"
                shift
                ;;
            *)
                echo "Unknown option or experiment type: $1"
                print_usage
                exit 1
                ;;
        esac
    done
    
    # Default to help if no experiment type specified
    if [[ -z "$experiment_type" && "$mode" != "help" ]]; then
        echo "Error: experiment type required"
        print_usage
        exit 1
    fi
    
    case "$mode" in
        list)
            list_experiments "$experiment_type"
            ;;
        submit)
            submit_experiments "$experiment_type"
            ;;
        local)
            run_local "$experiment_type"
            ;;
        help)
            print_usage
            ;;
    esac
}

main "$@"
