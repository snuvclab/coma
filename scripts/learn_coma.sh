skip_done=true

# Function to parse arguments and set values
parse_args() {
  while [[ $# -gt 0 ]]; do
    case $1 in
      --IoU_threshold_min)
        IoU_threshold_min="$2"
        shift 2
        ;;
      --inlier_num_threshold_min)
        inlier_num_threshold_min="$2"
        shift 2
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
    python src/coma/filter.py --IoU_threshold_min $IoU_threshold_min --inlier_num_threshold_min $inlier_num_threshold_min --supercategories $supercategory --categories $category --skip_done
    python src/coma/downsample_human.py --skip_done

    python src/coma/downsample_objects.py --dataset_type $dataset_type --supercategories $supercategory --categories $category --number_of_points 2048 --skip_done
    python src/coma/downsample_objects.py --dataset_type $dataset_type --supercategories $supercategory --categories $category --number_of_points 1500 --skip_done
    python src/coma/downsample_objects.py --dataset_type $dataset_type --supercategories $supercategory --categories $category --number_of_points 180 --skip_done

    python src/coma/extract_coma.py --supercategories $supercategory --categories $category --hyperparams_key "qual:${category}_object" --skip_done
    python src/coma/extract_coma.py --supercategories $supercategory --categories $category --hyperparams_key "qual:${category}_human" --skip_done
    python src/coma/extract_coma.py --supercategories $supercategory --categories $category --hyperparams_key "qual:${category}_occupancy" --skip_done
else
    python src/coma/filter.py --IoU_threshold_min $IoU_threshold_min --inlier_num_threshold_min $inlier_num_threshold_min --supercategories $supercategory --categories $category
    python src/coma/downsample_human.py

    python src/coma/downsample_objects.py --dataset_type $dataset_type --supercategories $supercategory --categories $category --number_of_points 2048
    python src/coma/downsample_objects.py --dataset_type $dataset_type --supercategories $supercategory --categories $category --number_of_points 1500
    python src/coma/downsample_objects.py --dataset_type $dataset_type --supercategories $supercategory --categories $category --number_of_points 180

    python src/coma/extract_coma.py --supercategories $supercategory --categories $category --hyperparams_key "qual:${category}_object"
    python src/coma/extract_coma.py --supercategories $supercategory --categories $category --hyperparams_key "qual:${category}_human"
    python src/coma/extract_coma.py --supercategories $supercategory --categories $category --hyperparams_key "qual:${category}_occupancy"
fi
