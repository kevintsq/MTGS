#!/usr/bin/env bash

set -e  # Exit immediately if any command fails

# Default values
CONFIG=""
NUM_WORKERS=12
NUM_GPUS=1
EXPORT_DIR=""
OUTPUT_DIR=""

# Parse long options
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --export-dir)
            EXPORT_DIR="$2"
            shift 2
            ;;
        --workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --config CONFIG --export-dir DIR --output-dir DIR [--workers NUM] [--gpus NUM]"
            echo "  --config:     Configuration file path (required)"
            echo "  --output-dir: Output directory path (required)"
            echo "  --export-dir: Export directory path (required)"
            echo "  --workers:    Number of workers (default: 12)"
            echo "  --gpus:       Number of GPUs (default: 1)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$CONFIG" ] || [ -z "$EXPORT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: --config, --export-dir and --output-dir are required"
    echo "Usage: $0 --config CONFIG --export-dir DIR --output-dir DIR [--workers NUM] [--gpus NUM]"
    exit 1
fi

# On failure keep the output directory in place: the preprocessing steps are
# resumable (masks/depth skip existing files), so a retry after a transient
# error continues instead of starting over. (Upstream moved it to *_failed.)
cleanup_on_failure() {
    echo "[ERROR] Command failed - keeping $OUTPUT_DIR for resume"
}

# Set trap to run cleanup on script failure
trap cleanup_on_failure ERR

echo "Running with:
    CONFIG=$CONFIG
    OUTPUT_DIR=$OUTPUT_DIR
    EXPORT_DIR=$EXPORT_DIR
    WORKERS=$NUM_WORKERS
    GPUS=$NUM_GPUS"

# Automated stand-in for the manual preview step: discover all in-block
# traversals (db-only) and cap selected_videos at the official 6 before the
# heavy preprocessing. No-op if the config already has a selection.
# Set MTGS_GROUPS_JSON to prioritise the location's navtest logs.
python -m nuplan_scripts.auto_reconstruction.auto_select_traversals \
    --config "$CONFIG" --num_workers "$NUM_WORKERS"

bash nuplan_scripts/preprocess.sh "$CONFIG" "$NUM_WORKERS" "$NUM_GPUS"
python -m nuplan_scripts.auto_reconstruction.background_reconstruction --config "$CONFIG"
python -m nuplan_scripts.auto_reconstruction.render_reconstruction --config "$CONFIG"
python -m nuplan_scripts.auto_reconstruction.export_reconstruction --config "$CONFIG" --output_dir "$EXPORT_DIR"
python -m nuplan_scripts.clean_temp_files --config "$CONFIG"

# If we reach here, all commands succeeded - disable cleanup trap
trap - ERR

echo "All commands completed successfully" 