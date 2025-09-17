#!/bin/bash

# run_full_3d_editing_pipeline.sh
# Usage: ./run_full_3d_editing_pipeline.sh "<prompt>"

set -e

# Function to print usage
usage() {
    echo "Usage: $0 \"<prompt>\""
    echo "  <prompt> : Text prompt for inpainting step (required)"
    exit 1
}

# Check for prompt argument
if [ $# -ne 1 ]; then
    usage
fi

PROMPT="$1"

# Check required files and directories
DIR_NAME="${DIR_NAME:-example}"
SOURCE_MODEL="assets/${DIR_NAME}/model.glb"
MASK_MODEL="assets/${DIR_NAME}/mask.glb"
OUTPUT_DIR="outputs"
IMAGE_DIR="$OUTPUT_DIR/images"
RENDER_IMAGE="$IMAGE_DIR/render_0002.png"
MASK_IMAGE="$IMAGE_DIR/mask_0002.png"

if [ ! -f "$SOURCE_MODEL" ]; then
    echo "Error: Source model not found at $SOURCE_MODEL"
    exit 2
fi

if [ ! -f "$MASK_MODEL" ]; then
    echo "Error: Mask model not found at $MASK_MODEL"
    exit 2
fi

# Step 1: Render RGB and mask images
echo "Rendering RGB and mask images..."
python utils/render_rgb_and_mask.py \
    --source_model "$SOURCE_MODEL" \
    --mask_model "$MASK_MODEL" \
    --output_dir "$OUTPUT_DIR"

# Verify render output
if [ ! -f "$RENDER_IMAGE" ]; then
    echo "Error: Rendered image not found at $RENDER_IMAGE"
    exit 3
fi

if [ ! -f "$MASK_IMAGE" ]; then
    echo "Error: Mask image not found at $MASK_IMAGE"
    exit 3
fi

# Step 2: Inpaint the rendered image
echo "Running inpainting..."
python utils/inpaint.py \
    --image_path "$RENDER_IMAGE" \
    --mask_path "$MASK_IMAGE" \
    --output_dir "$IMAGE_DIR" \
    --prompt "$PROMPT"

# Step 3: Run inference
echo "Running inference..."
python inference.py \
    --input_model "$SOURCE_MODEL" \
    --mask_model "$MASK_MODEL" \
    --image_dir "$IMAGE_DIR" \
    --output_dir "$OUTPUT_DIR"

echo "Pipeline completed successfully."