#!/bin/bash

OUTPATH=/lpai/output/models/open_clip1e3/downstreams/cls_zeroshoshot-imgsketch
BS=128
MODEL="ViT-B-16"
#MODEL="ViT-B-16-512"
#MODEL="ViT-L-16"

CHKPNT=/lpai/models/openclip/vitblr1e3/open_clip/clip-vitb16-cc12m-epochs30/checkpoints/epoch_30.pt

SKETCH=/lpai/volumes/so-volume-bd-ga/lhp/datasets/imagenetsketch/data/sketch
# V2=path/to/datasets/imagenetv2-matched-frequency-format-val
# A=path/to/datasets/imagenet-a
# O=path/to/datasets/imagenet-o
# R=path/to/datasets/imagenet_r/imagenet-r

CUDA_VISIBLE_DEVICES=5 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-sketch=$SKETCH
# python -m main.run --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-a=$A
# python -m main.run --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-o=$O
# python -m main.run --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-r=$R
# python -m main.run --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-v2=$V2
