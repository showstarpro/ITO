#!/usr/bin/env bash
# ITO ViT-B/16 pre-training on CC12M (8 GPUs x 256 = global batch size 2048).
#
# Usage:
#   CC12M_DIR=/path/to/cc12m IMAGENET_VAL=/path/to/imagenet/val bash scripts/ito/train_cc12m.sh

set -e
cd "$(dirname "$0")/../../src"

CC12M_DIR=${CC12M_DIR:-/path/to/cc12m_webdataset}
IMAGENET_VAL=${IMAGENET_VAL:-/path/to/imagenet/val}

torchrun --nproc_per_node ${NPROC_PER_NODE:-8} -m open_clip_train.main \
    --train-data "${CC12M_DIR}/cc12m-train-{0000..2175}.tar" \
    --train-num-samples 10968539 \
    --imagenet-val "${IMAGENET_VAL}" \
    --dataset-type webdataset \
    --batch-size 256 \
    --precision amp \
    --workers 16 \
    --lr 1e-3 --wd 0.1 --warmup 10000 --beta1 0.9 --beta2 0.98 --eps 1e-06 \
    --epochs 30 \
    --model ViT-B-16 \
    --coca-caption-loss-weight 0 --coca-contrastive-loss-weight 0 \
    --alpha 2 \
    --aug \
    --aug-cfg gray_scale_prob=0.2 color_jitter=0.4,0.4,0.4,0.1 color_jitter_prob=0.8 scale=0.5,1.0 \
    --name ito_cc12m_vitb16 \
    "$@"
