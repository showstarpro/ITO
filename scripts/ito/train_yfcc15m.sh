#!/usr/bin/env bash
# ITO ViT-B/16 pre-training on YFCC15M.
# Global batch size 4096 (e.g. 16 GPUs x 256). Adjust --batch-size to your GPU count.
#
# Usage (single node, 8 GPUs):
#   YFCC15M_DIR=/path/to/yfcc15m IMAGENET_VAL=/path/to/imagenet/val bash scripts/ito/train_yfcc15m.sh
# Multi-node: set NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT.

set -e
cd "$(dirname "$0")/../../src"

YFCC15M_DIR=${YFCC15M_DIR:-/path/to/yfcc15m_webdataset}
IMAGENET_VAL=${IMAGENET_VAL:-/path/to/imagenet/val}
SENTENCE_POOL=${SENTENCE_POOL:-cls}   # cls | mean | token
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

torchrun \
    --nnodes ${NNODES:-1} --node_rank ${NODE_RANK:-0} --nproc_per_node ${NPROC_PER_NODE} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} --master_port ${MASTER_PORT:-29500} \
    -m open_clip_train.main \
    --train-data "${YFCC15M_DIR}/0{0000..1410}.tar" \
    --train-num-samples 14082031 \
    --imagenet-val "${IMAGENET_VAL}" \
    --dataset-type webdataset \
    --batch-size 256 \
    --precision amp \
    --workers 16 \
    --lr 3e-3 --wd 0.1 --warmup 10000 --beta1 0.9 --beta2 0.98 --eps 1e-06 \
    --epochs 30 \
    --model ViT-B-16 \
    --coca-caption-loss-weight 0 --coca-contrastive-loss-weight 0 \
    --alpha 2 \
    --sentence-pool ${SENTENCE_POOL} \
    --aug \
    --aug-cfg scale=0.5,1.0 gray_scale_prob=0.2 color_jitter=0.4,0.4,0.4,0.1 color_jitter_prob=0.8 \
    --name ito_yfcc15m_vitb16_${SENTENCE_POOL} \
    "$@"
