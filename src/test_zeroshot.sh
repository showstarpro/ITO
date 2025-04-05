#!/bin/bash

OUTPATH=/lpai/output/models/siglip12m1e3/downstreams/zeroshot-cifar100
BS=4096
MODEL="ViT-B-16-SigLIP"
#MODEL="ViT-B-16-512"
#MODEL="ViT-L-16"

# CHKPNT=/lpai/models/openclip/vitblr1e3/open_clip/clip-vitb16-cc12m-epochs30/checkpoints/epoch_30.pt
# CHKPNT=/lpai/models/sentence-clip/siglip3m1e3/siglip3m_1e3/checkpoints/epoch_30.pt
CHKPNT=/lpai/models/sentence-clip/siglip12m1e3/siglip12m_1e3/checkpoints/epoch_30.pt

# CIFAR10=/lpai/volumes/so-volume-bd-ga/lhp/datasets/cifar10
CIFAR100=/lpai/volumes/so-volume-bd-ga/lhp/datasets/cifar100
# IMAGENETVAL=/lpai/dataset/imagenet-1k/0-1-0/ILSVRC2012/val
# IMAGENETTRAIN=/lpai/dataset/imagenet-1k/0-1-0/train
# FLOWERS102=/lpai/volumes/so-volume-bd-ga/lhp/datasets/flower_102/dataset
# FOOD101=/lpai/volumes/so-volume-bd-ga/lhp/datasets/food_101
# STANFORD=/lpai/volumes/so-volume-bd-ga/lhp/datasets

# python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-val=$IMAGENETVAL
# CUDA_VISIBLE_DEVICES=4 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --cifar10=$CIFAR10
CUDA_VISIBLE_DEVICES=4 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --cifar100=$CIFAR100
# CUDA_VISIBLE_DEVICES=1 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-val=$IMAGENETVAL --imagenet-train=$IMAGENETTRAIN
# CUDA_VISIBLE_DEVICES=4 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --flowers-102=$FLOWERS102
# python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --food-101=$FOOD101 
# CUDA_VISIBLE_DEVICES=6 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --stanford=$STANFORD 
