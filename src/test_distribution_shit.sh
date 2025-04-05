#!/bin/bash

OUTPATH=/lpai/output/models/siglip3m1e3/downstreams/cls_zeroshoshot-img-sketch
BS=128
MODEL="ViT-B-16-SigLIP"
#MODEL="ViT-B-16-512"
#MODEL="ViT-L-16"

CHKPNT=/lpai/models/sentence-clip/siglip3m1e3/siglip3m_1e3/checkpoints/epoch_30.pt
# CHKPNT=/lpai/models/sentence-clip/siglip12m1e3/siglip12m_1e3/checkpoints/epoch_30.pt

SKETCH=/lpai/volumes/so-volume-bd-ga/lhp/datasets/imagenetsketch/data/sketch
# V2=path/to/datasets/imagenetv2-matched-frequency-format-val
# A=/lpai/volumes/so-volume-bd-ga/lhp/datasets/imagenet-a/test_data
# O=path/to/datasets/imagenet-o
# R=/lpai/volumes/so-volume-bd-ga/lhp/datasets/imagenet-r/test_data

CUDA_VISIBLE_DEVICES=5 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-sketch=$SKETCH
# CUDA_VISIBLE_DEVICES=5 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-a=$A
# python -m main.run --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-o=$O
# CUDA_VISIBLE_DEVICES=5 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-r=$R
# python -m main.run --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-v2=$V2
