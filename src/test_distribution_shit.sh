#!/bin/bash

OUTPATH=/lpai/output/models/openclipcc12m/downstreams/cls_zeroshoshot-img-r
BS=128
MODEL="ViT-B-16"
#MODEL="ViT-B-16-512"
#MODEL="ViT-L-16"

CHKPNT=/lpai/models/openclip/vitblr1e3/open_clip/clip-vitb16-cc12m-epochs30/checkpoints/epoch_30.pt
# CHKPNT=/lpai/models/openclip/openclipcc3m/open_clip_cc3m_1/checkpoints/epoch_30.pt

# SKETCH=/lpai/volumes/so-volume-bd-ga/lhp/datasets/imagenetsketch/data/sketch
# V2=path/to/datasets/imagenetv2-matched-frequency-format-val
# A=/lpai/volumes/so-volume-bd-ga/lhp/datasets/imagenet-a/test_data
# O=path/to/datasets/imagenet-o
R=/lpai/volumes/so-volume-bd-ga/lhp/datasets/imagenet-r/test_data

# CUDA_VISIBLE_DEVICES=5 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-sketch=$SKETCH
# CUDA_VISIBLE_DEVICES=5 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-a=$A
# python -m main.run --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-o=$O
CUDA_VISIBLE_DEVICES=5 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-r=$R
# python -m main.run --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-v2=$V2
