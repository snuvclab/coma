#!/bin/bash

### Define a list of GPU IDs you want to use for parallel run ###
gpu_ids=(0 1 2 3 4 5 6 7)

# Extract the values from the captured output and set default variables
supercategories=()
categories=()
prompts=()

# Default variables for 'action=store_true' flags
verbose=false
skip_done=true ## note!!
# free_dimension_constraint=false
disable_lowres_switch_for_behave=false
enable_aggregate_total_prompts=false

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
      --prompts)
        shift # Consume the --prompts flag
        while [[ $# -gt 0 ]]; do
          if [[ $1 == --* ]]; then
            break
          fi
          prompts+=("$1")
          shift
        done
        ;;
      --camera_dir)
        camera_dir="$2"
        shift 2
        ;;
      --human_after_opt_dir)
        human_after_opt_dir="$2"
        shift 2
        ;;
      --human_pred_dir)
        human_pred_dir="$2"
        shift 2
        ;;
      --save_dir)
        save_dir="$2"
        shift 2
        ;;
      --enable_aggregate_total_prompts)
        enable_aggregate_total_prompts=true
        shift 1
        ;;
      --disable_lowres_switch_for_behave)
        disable_lowres_switch_for_behave=true
        shift 1
        ;;
      --no_skip_done)
        skip_done=false
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
command_to_run="blenderproc run src/generation/compute_metrics.py"

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
    --parallel_num $num_gpus"

  if [ "$enable_aggregate_total_prompts" == true ]; then
    full_command="$full_command --enable_aggregate_total_prompts"
  fi
  if [ "$disable_lowres_switch_for_behave" == true ]; then
    full_command="$full_command --disable_lowres_switch_for_behave"
  fi

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
  if [ ${#prompts[@]} -gt 0 ]; then
    full_command="$full_command --prompts"
    for prompt in "${prompts[@]}"; do
      full_command="$full_command '$prompt'"
    done
  fi
  if [ $human_after_opt_dir ]; then
    full_command="$full_command --human_after_opt_dir $human_after_opt_dir"
  fi
  if [ $human_pred_dir ]; then
    full_command="$full_command --human_pred_dir $human_pred_dir"
  fi
  if [ $save_dir ]; then
    full_command="$full_command --save_dir $save_dir"
  fi
  if [ "$skip_done" = true ]; then
    full_command="$full_command --skip_done"
  fi
  echo "$full_command"
  ## Run command with parallelism in multiple GPUs (ampersand is used for denoting concurrent command run!)
  eval $full_command &
done

# Wait for all background processes to finish
wait

