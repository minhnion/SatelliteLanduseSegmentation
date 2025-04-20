#!/bin/bash

INPUT="inference_tif"
OUTPUT="inference_png"
PRETRAINED_MODEL="inference_model/model.pth"
MODEL="Foundation"

python infer.py --input "$INPUT" --output "$OUTPUT" --pretrained "$PRETRAINED_MODEL"
