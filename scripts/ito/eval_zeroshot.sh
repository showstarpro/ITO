#!/usr/bin/env bash
# Zero-shot ImageNet evaluation of a trained checkpoint.
#
# Usage:
#   bash scripts/ito/eval_zeroshot.sh /path/to/epoch_30.pt /path/to/imagenet/val [MODEL]
# Set SENTENCE_POOL=mean/token when evaluating a checkpoint trained with that option.

set -e
cd "$(dirname "$0")/../../src"

CKPT=$1
IMAGENET_VAL=$2
MODEL=${3:-ViT-B-16}

python -m open_clip_train.main \
    --model ${MODEL} \
    --pretrained "${CKPT}" \
    --imagenet-val "${IMAGENET_VAL}" \
    --sentence-pool ${SENTENCE_POOL:-cls}
