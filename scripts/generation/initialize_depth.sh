#!/bin/bash

### Define a list of GPU IDs you want to use for parallel run ###
gpu_ids=(0 1 2 3 4 5 6 7)

# Extract the values from the captured output and set default variables
supercategories=()
categories=()
prompts=()

# Default variables for 'action=store_true' flags
skip_done=true ## note!!
verbose=false
disable_lowres_switch_for_behave=false
no_initialize=false

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
      --inpaint_dir)
        inpaint_dir="$2"
        shift 2
        ;;
      --human_pred_dir)
        human_pred_dir="$2"
        shift 2
        ;;
      --human_seg_dir)
        human_seg_dir="$2"
        shift 2
        ;;
      --camera_dir)
        camera_dir="$2"
        shift 2
        ;;
      --save_dir)
        save_dir="$2"
        shift 2
        ;;
      --interval_ratio)
        interval_ratio="$2"
        shift 2
        ;;
      --retrieval_range)
        retrieval_range="$2"
        shift 2
        ;;
      --kernel_size)
        kernel_size="$2"
        shift 2
        ;;
      --max_collisions)
        max_collisions="$2"
        shift 2
        ;;
      --disable_lowres_switch_for_behave)
        disable_lowres_switch_for_behave=true
        shift 1
        ;;
      --no_initialize)
        no_initialize=true
        shift 1
        ;;
      --no_skip_done)
        skip_done=false
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
command_to_run="blenderproc run src/generation/initialize_depth.py"

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

  if [ "$disable_lowres_switch_for_behave" = true ]; then
    full_command="$full_command --disable_lowres_switch_for_behave"
  fi
  if [ "$no_initialize" = true ]; then
    full_command="$full_command --no_initialize"
  fi
  if [ "$skip_done" = true ]; then
    full_command="$full_command --skip_done"
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
  if [ ${#prompts[@]} -gt 0 ]; then
    full_command="$full_command --prompts"
    for prompt in "${prompts[@]}"; do
      full_command="$full_command '$prompt'"
    done
  fi

  if [ $inpaint_dir ]; then
    full_command="$full_command --inpaint_dir $inpaint_dir"
  fi
  if [ $human_pred_dir ]; then
    full_command="$full_command --human_pred_dir $human_pred_dir"
  fi
  if [ $human_seg_dir ]; then
    full_command="$full_command --human_seg_dir $human_seg_dir"
  fi
  if [ $camera_dir ]; then
    full_command="$full_command --camera_dir $camera_dir"
  fi
  if [ $save_dir ]; then
    full_command="$full_command --save_dir $save_dir"
  fi
  if [ $interval_ratio ]; then
    full_command="$full_command --interval_ratio $interval_ratio"
  fi
  if [ $retrieval_range ]; then
    full_command="$full_command --retrieval_range $retrieval_range"
  fi
  if [ $kernel_size ]; then
    full_command="$full_command --kernel_size $kernel_size"
  fi
  if [ $max_collisions ]; then
    full_command="$full_command --max_collisions $max_collisions"
  fi

  echo "$full_command"
  ## Run command with parallelism in multiple GPUs (ampersand is used for denoting concurrent command run!)
  eval $full_command &
done

# Wait for all background processes to finish
wait

