gpu_ids=(0 1 2 3 4 5 6 7)
skip_done=true

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
      --dataset_type)
        dataset_type="$2"
        shift 2
        ;;
      --supercategory)
        supercategory="$2"
        shift 2
        ;;
      --category)
        category="$2"
        shift 2
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

if [ "$skip_done" = true ]; then
    blenderproc run src/generation/render_objects.py --dataset_types $dataset_type --supercategories $supercategory --categories $category --skip_done
    python src/generation/select_mask.py --supercategories $supercategory --categories $category --skip_done
    python src/generation/generate_prompts.py --supercategories $supercategory --categories $category --skip_done
    bash scripts/generation/inpaint.sh --supercategories $supercategory --categories $category --gpus ${gpu_ids[@]}
else
    blenderproc run src/generation/render_objects.py --dataset_types $dataset_type --supercategories $supercategory --categories $category
    python src/generation/select_mask.py --supercategories $supercategory --categories $category
    python src/generation/generate_prompts.py --supercategories $supercategory --categories $category
    bash scripts/generation/inpaint.sh --supercategories $supercategory --categories $category --gpus ${gpu_ids[@]} --no_skip_done
fi
