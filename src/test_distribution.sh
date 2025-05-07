#!/bin/bash

OUTPATH=/path/to/output/models/downstreams/distribution
BS=128
MODEL="ViT-B-16"
# #MODEL="ViT-B-16-512"
# #MODEL="ViT-L-16"

CHKPNT=/path/to/models/checkpoints/epoch_30.pt

SKETCH=/path/to/datasets/sketch
A=/path/to/datasets/imagenet-a/test_data
R=/path/to/datasets/imagenet-r/test_data

python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-sketch=$SKETCH
# python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-a=$A
# python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-r=$R