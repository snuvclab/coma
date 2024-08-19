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
      --inpaint_dir)
        inpaint_dir="$2"
        shift 2
        ;;
      --asset_seg_dir)
        asset_seg_dir="$2"
        shift 2
        ;;
      --human_prefilter_dir)
        human_prefilter_dir="$2"
        shift 2
        ;;
      --human_initial_dir)
        human_initial_dir="$2"
        shift 2
        ;;
      --human_preds_dir)
        human_preds_dir="$2"
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
      --smplx_path)
        smplx_path="$2"
        shift 2
        ;;
      --maximum_candidates)
        maximum_candidates="$2"
        shift 2
        ;;
      --ransac_threshold)
        ransac_threshold="$2"
        shift 2
        ;;
      --triangulation_threshold)
        triangulation_threshold="$2"
        shift 2
        ;;
      --num_epoch)
        num_epoch="$2"
        shift 2
        ;;
      --minimum_inliers)
        minimum_inliers="$2"
        shift 2
        ;;
      --lr)
        lr="$2"
        shift 2
        ;;
      --w_collision)
        w_collision="$2"
        shift 2
        ;;
      --w_multiview)
        w_multiview="$2"
        shift 2
        ;;
      --w_refview)
        w_refview="$2"
        shift 2
        ;;
      --seed)
        seed="$2"
        shift 2
        ;;
      --disable_lowres_switch_for_behave)
        disable_lowres_switch_for_behave=true
        shift 1
        ;;
      --enable_aggregate_total_prompts)
        enable_aggregate_total_prompts=true
        shift 1
        ;;
      --no_initialize)
        no_initialize=true
        shift 1
        ;;
      --no_collision)
        no_collision=true
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
command_to_run="python src/generation/optimize_depth.py"

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
  if [ "$enable_aggregate_total_prompts" == true ]; then
    full_command="$full_command --enable_aggregate_total_prompts"
  fi
  if [ "$no_initialize" = true ]; then
    full_command="$full_command --no_initialize"
  fi
  if [ "$no_collision" = true ]; then
    full_command="$full_command --no_collision"
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
  if [ $asset_seg_dir ]; then
    full_command="$full_command --asset_seg_dir $asset_seg_dir"
  fi
  if [ $human_prefilter_dir ]; then
    full_command="$full_command --human_prefilter_dir $human_prefilter_dir"
  fi
  if [ $human_initial_dir ]; then
    full_command="$full_command --human_initial_dir $human_initial_dir"
  fi
  if [ $human_preds_dir ]; then
    full_command="$full_command --human_preds_dir $human_preds_dir"
  fi
  if [ $camera_dir ]; then
    full_command="$full_command --camera_dir $camera_dir"
  fi
  if [ $save_dir ]; then
    full_command="$full_command --save_dir $save_dir"
  fi
  if [ $smplx_path ]; then
    full_command="$full_command --smplx_path $smplx_path"
  fi
  if [ $maximum_candidates ]; then
    full_command="$full_command --maximum_candidates $maximum_candidates"
  fi
  if [ $ransac_threshold ]; then
    full_command="$full_command --ransac_threshold $ransac_threshold"
  fi
  if [ $triangulation_threshold ]; then
    full_command="$full_command --triangulation_threshold $triangulation_threshold"
  fi
  if [ $num_epoch ]; then
    full_command="$full_command --num_epoch $num_epoch"
  fi
  if [ $minimum_inliers ]; then
    full_command="$full_command --minimum_inliers $minimum_inliers"
  fi
  if [ $lr ]; then
    full_command="$full_command --lr $lr"
  fi
  if [ $w_collision ]; then
    full_command="$full_command --w_collision $w_collision"
  fi
  if [ $w_multiview ]; then
    full_command="$full_command --w_multiview $w_multiview"
  fi
  if [ $w_refview ]; then
    full_command="$full_command --w_refview $w_refview"
  fi

  echo "$full_command"
  ## Run command with parallelism in multiple GPUs (ampersand is used for denoting concurrent command run!)
  eval $full_command &
done

# Wait for all background processes to finish
wait

