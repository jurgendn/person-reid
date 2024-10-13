#!/bin/bash

# Default values for optional parameters
config_file="configs/pose_hrnet.yaml"
pretrained="pretrained/hrnet_w32_256x192.pth"

# Parse flags using getopts
while [[ "$#" -gt 0 ]]; do
    case $1 in
    --metadata)
        metadata="$2"
        shift
        ;;
    --dataset-name)
        dataset_name="$2"
        shift
        ;;
    --target-set)
        target_set="$2"
        shift
        ;;
    --batch-size)
        batch_size="$2"
        shift
        ;;
    --device)
        device="$2"
        shift
        ;;
    --config-file)
        config_file="$2"
        shift
        ;; # Optional
    --pretrained)
        pretrained="$2"
        shift
        ;; # Optional
    --help)
        echo "Usage: $0 --metadata <metadata> --dataset-name <dataset_name> --target-set <target_set> --batch-size <batch_size> --device <device> [--config-file <config_file>] [--pretrained <pretrained>]"
        exit 0
        ;;
    *)
        echo "Unknown parameter: $1"
        echo "Use --help for usage."
        exit 1
        ;;
    esac
    shift
done

# Check for required parameters
if [[ -z "$metadata" ]]; then
    echo "Error: --metadata is required"
    exit 1
fi

if [[ -z "$dataset_name" ]]; then
    echo "Error: --dataset-name is required"
    exit 1
fi

if [[ -z "$target_set" ]]; then
    echo "Error: --target-set is required"
    exit 1
fi

if [[ -z "$batch_size" ]]; then
    echo "Error: --batch-size is required"
    exit 1
fi

if [[ -z "$device" ]]; then
    echo "Error: --device is required"
    exit 1
fi

# Run the Python script
PYTHONPATH=. uv run tools/get_pose.py \
    --metadata "$metadata" \
    --dataset-name "$dataset_name" \
    --target-set "$target_set" \
    --batch-size "$batch_size" \
    --device "$device" \
    --config-file "$config_file" \
    --pretrained "$pretrained"
