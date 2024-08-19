#!/bin/bash

### Define a list of GPU IDs you want to use for parallel run ###
gpu_ids=(0 1 2 3 4 5 6 7)

### Define default values for arguments ###
# Execute a Python script to retrieve values from src.generation.inpaint.py
source_constants=$(python -c "HYPERPARAMS = __import__('src.generation.inpaint', fromlist=[ \
    'NUM_IMG_PER_COMBINATION', \
    'PROMPTS_DIR', \
    'ASSET_RENDER_DIR', \
    'ASSET_MASK_DIR', \
    'ASSET_SEG_DIR', \
    'SAVE_DIR', \
    'LDM_MODEL_KEY', \
    'ADAPTIVE_MASK_MODEL_TYPE', \
    'DEFAULT_CFG_SCALE', \
    'DEFAULT_STRENGTH', \
    'DEFAULT_DDIM_STEPS', \
    'DEFAULT_POINTREND_THRESHOLD', \
    'DEFAULT_ENFORCE_FULL_MASK_RATIO', \
    'DEFAULT_HUMAN_DETECTION_THRES', \
    'NEGATIVE_PROMPT', \
    'SKIP_DONE',]);  \
    print(HYPERPARAMS.NUM_IMG_PER_COMBINATION); \
    print(HYPERPARAMS.PROMPTS_DIR); \
    print(HYPERPARAMS.ASSET_RENDER_DIR); \
    print(HYPERPARAMS.ASSET_MASK_DIR); \
    print(HYPERPARAMS.ASSET_SEG_DIR); \
    print(HYPERPARAMS.SAVE_DIR); \
    print(HYPERPARAMS.LDM_MODEL_KEY); \
    print(HYPERPARAMS.ADAPTIVE_MASK_MODEL_TYPE); \
    print(HYPERPARAMS.DEFAULT_CFG_SCALE); \
    print(HYPERPARAMS.DEFAULT_STRENGTH); \
    print(HYPERPARAMS.DEFAULT_DDIM_STEPS); \
    print(HYPERPARAMS.DEFAULT_POINTREND_THRESHOLD); \
    print(HYPERPARAMS.DEFAULT_ENFORCE_FULL_MASK_RATIO); \
    print(HYPERPARAMS.DEFAULT_HUMAN_DETECTION_THRES); \
    print(HYPERPARAMS.NEGATIVE_PROMPT); \
    from constants.metadata import DEFAULT_SEED; \
    print(DEFAULT_SEED);"
)

# Extract the values from the captured output and set default variables
supercategories=()
categories=()
num_img_per_combination=$(echo "$source_constants" | sed -n '1p')
prompts_dir=$(echo "$source_constants" | sed -n '2p')
asset_render_dir=$(echo "$source_constants" | sed -n '3p')
asset_mask_dir=$(echo "$source_constants" | sed -n '4p')
asset_seg_dir=$(echo "$source_constants" | sed -n '5p')
save_dir=$(echo "$source_constants" | sed -n '6p')
ldm_model_key=$(echo "$source_constants" | sed -n '7p')
adaptive_mask_model_type=$(echo "$source_constants" | sed -n '8p')
default_cfg_scale=$(echo "$source_constants" | sed -n '9p')
default_strength=$(echo "$source_constants" | sed -n '10p')
default_ddim_steps=$(echo "$source_constants" | sed -n '11p')
default_pointrend_threshold=$(echo "$source_constants" | sed -n '12p')
default_enforce_full_mask_ratio=$(echo "$source_constants" | sed -n '13p')
default_human_detection_thres=$(echo "$source_constants" | sed -n '14p')
negative_prompt=$(echo "$source_constants" | sed -n '15p')
seed=$(echo "$source_constants" | sed -n '16p')

# Default variables for 'action=store_true' flags
enable_sam_multitask_output=false
enable_safety_checker=false
use_visualizer=false
skip_done=true ## note!!
verbose=false

# Function to parse arguments and set values
parse_args() {
  while [[ $# -gt 0 ]]; do
    case $1 in
      --gpus)
        shift # Consume the --gpus flag
        gpu_ids=() # Reset the 'gpu_ids'
        while [[ $# -gt 0 ]]; do
          if [[ $1 == --* ]]; then
            break
          fi
          gpu_ids+=("$1")
          shift
        done
        ;;
      --supercategories)
        shift # Consume the --supercategories flag
        while [[ $# -gt 0 ]]; do
          if [[ $1 == --* ]]; then
            break
          fi
          supercategories+=("$1")
          shift
        done
        ;;
      --categories)
        shift # Consume the --categories flag
        while [[ $# -gt 0 ]]; do
          if [[ $1 == --* ]]; then
            break
          fi
          categories+=("$1")
          shift
        done
        ;;
      --num_img_per_combination)
        num_img_per_combination="$2"
        shift 2
        ;;
      --prompts_dir)
        prompts_dir="$2"
        shift 2
        ;;
      --asset_render_dir)
        asset_render_dir="$2"
        shift 2
        ;;
      --asset_mask_dir)
        asset_mask_dir="$2"
        shift 2
        ;;
      --asset_seg_dir)
        asset_seg_dir="$2"
        shift 2
        ;;
      --save_dir)
        save_dir="$2"
        shift 2
        ;;
      --ldm_model_key)
        ldm_model_key="$2"
        shift 2
        ;;
      --adaptive_mask_model_type)
        adaptive_mask_model_type="$2"
        shift 2
        ;;
      --default_cfg_scale)
        default_cfg_scale="$2"
        shift 2
        ;;
      --default_strength)
        default_strength="$2"
        shift 2
        ;;
      --default_ddim_steps)
        default_ddim_steps="$2"
        shift 2
        ;;
      --default_pointrend_threshold)
        default_pointrend_threshold="$2"
        shift 2
        ;;
      --default_enforce_full_mask_ratio)
        default_enforce_full_mask_ratio="$2"
        shift 2
        ;;
      --default_human_detection_thres)
        default_human_detection_thres="$2"
        shift 2
        ;;
      --negative_prompt)
        negative_prompt="$2"
        shift 2
        ;;
      --seed)
        seed="$2"
        shift 2
        ;;
      --no_skip_done)
        skip_done=false
        shift 1
        ;;
      --enable_sam_multitask_output)
        enable_sam_multitask_output=true
        shift 1
        ;;
      --enable_safety_checker)
        enable_safety_checker=true
        shift 1
        ;;
      --use_visualizer)
        use_visualizer=true
        shift 1
        ;;
      --verbose)
        verbose=true
        shift 1
        ;;
      *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
  done
}

# Call the function to parse command-line arguments
parse_args "$@"

# Define your command to run on each GPU
command_to_run="python src/generation/inpaint.py"

# Loop through GPU IDs and run the command on each GPU with arguments
num_gpus=${#gpu_ids[@]}
echo "You are using: $num_gpus GPUs."
for index in "${!gpu_ids[@]}"; do
  gpu_id=${gpu_ids[$index]}
  echo "Running [Split $index] on: [GPU $gpu_id]"
  # full command
  full_command="CUDA_VISIBLE_DEVICES=$gpu_id \
    $command_to_run \
    --parallel_idx $index \
    --parallel_num $num_gpus \
    --num_img_per_combination $num_img_per_combination \
    --prompts_dir '$prompts_dir' \
    --asset_render_dir '$asset_render_dir' \
    --asset_mask_dir '$asset_mask_dir' \
    --asset_seg_dir '$asset_seg_dir' \
    --save_dir '$save_dir' \
    --ldm_model_key '$ldm_model_key' \
    --adaptive_mask_model_type '$adaptive_mask_model_type' \
    --default_cfg_scale $default_cfg_scale \
    --default_strength $default_strength \
    --default_ddim_steps $default_ddim_steps \
    --default_pointrend_threshold $default_pointrend_threshold \
    --default_enforce_full_mask_ratio $default_enforce_full_mask_ratio \
    --default_human_detection_thres $default_human_detection_thres \
    --negative_prompt '$negative_prompt' \
    --seed $seed"
  
  # Check if 'action=store_true' flags are set
  if [ "$enable_sam_multitask_output" = true ]; then
    full_command="$full_command --enable_sam_multitask_output"
  fi
  if [ "$enable_safety_checker" = true ]; then
    full_command="$full_command --enable_safety_checker"
  fi
  if [ "$skip_done" = true ]; then
    full_command="$full_command --skip_done"
  fi
  if [ "$use_visualizer" = true ]; then
    full_command="$full_command --use_visualizer"
  fi
  if [ "$verbose" = true ]; then
    full_command="$full_command --verbose"
  fi

  # Check if supercategories/categories is provided and add the values if it is
  if [ ${#supercategories[@]} -gt 0 ]; then
    full_command="$full_command --supercategories"
    for supercategory in "${supercategories[@]}"; do
      full_command="$full_command '$supercategory'"
    done
  fi
  if [ ${#categories[@]} -gt 0 ]; then
    full_command="$full_command --categories"
    for category in "${categories[@]}"; do
      full_command="$full_command '$category'"
    done
  fi

  ## Run command with parallelism in multiple GPUs (ampersand is used for denoting concurrent command run!)
  eval $full_command &
done

# Wait for all background processes to finish
wait