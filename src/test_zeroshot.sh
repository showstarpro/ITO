#!/bin/bash

# OUTPATH=/lpai/output/models/ito-cc12m-sub2lr1e3b2k/downstreams/zeroshot-cifar100
# BS=4096
# MODEL="ViT-B-16"
# #MODEL="ViT-B-16-512"
# #MODEL="ViT-L-16"

# # CHKPNT=/lpai/models/sentence-clip/itocc12m/sentence_img_aug_cc3m_a2_sc05/checkpoints/epoch_30.pt
# # CHKPNT=/lpai/models/sentence-clip/itocc3m/sentence_img_aug_cc3m_a2_sc05/checkpoints/epoch_30.pt
# # CHKPNT=/lpai/models/sentence-clip/la100mspsc09/sentence_img_aug_cc3m_a1/checkpoints/epoch_30.pt
# # CHKPNT=/lpai/models/ito/dm3mtxt2/ITO_reca_sub2_lr1e3_bs2048/checkpoints/epoch_30.pt
# # CHKPNT=/lpai/volumes/jfs-data-lhp-bd-ga/ito_dreamcc3m/epoch_30.pt
# # CHKPNT=/lpai/volumes/jfs-data-lhp-bd-ga/ito_recap/ITO_reca_sub3_lr1e3_bs2048/checkpoints/epoch_30.pt
# # CHKPNT=/lpai/volumes/jfs-data-lhp-bd-ga/ito_cc3m/sub2lr3e3bs4k/epoch_30.pt
# CHKPNT=/lpai/volumes/jfs-data-lhp-bd-ga/ito-cc12m/sub2lr1e3b2k/epoch_30.pt
# # CHKPNT=/lpai/volumes/jfs-data-lhp-bd-ga/ito-cc12m/sub2lr1e3b4k/epoch_30.pt

# CIFAR10=/lpai/volumes/so-volume-bd-ga/lhp/datasets/cifar10
# CIFAR100=/lpai/volumes/so-volume-bd-ga/lhp/datasets/cifar100
# IMAGENETVAL=/lpai/dataset/imagenet-1k/0-1-0/ILSVRC2012/val
# IMAGENETTRAIN=/lpai/dataset/imagenet-1k/0-1-0/train
# FLOWERS102=/lpai/volumes/so-volume-bd-ga/lhp/datasets/flower_102/dataset
# FOOD101=/lpai/volumes/so-volume-bd-ga/lhp/datasets/food_101
# PETS=/lpai/volumes/so-volume-bd-ga/lhp/datasets/pets
# STANFORD=/lpai/volumes/so-volume-bd-ga/lhp/datasets

# # python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-val=$IMAGENETVAL
# # CUDA_VISIBLE_DEVICES=4 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --cifar10=$CIFAR10
# CUDA_VISIBLE_DEVICES=4 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --cifar100=$CIFAR100
# # CUDA_VISIBLE_DEVICES=1 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-val=$IMAGENETVAL --imagenet-train=$IMAGENETTRAIN
# # CUDA_VISIBLE_DEVICES=4 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --flowers-102=$FLOWERS102
# # CUDA_VISIBLE_DEVICES=4 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --food-101=$FOOD101 
# # CUDA_VISIBLE_DEVICES=4 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --pets=$PETS
# # CUDA_VISIBLE_DEVICES=4 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --stanford=$STANFORD 


OUTPATH=/lpai/output/models/ito-cc12m-sub2lr1e3b2k/downstreams/cls_zeroshoshot-imgs-s
BS=128
MODEL="ViT-B-16"
#MODEL="ViT-B-16-512"
#MODEL="ViT-L-16"

# CHKPNT=/lpai/models/sentence-clip/itocc12m/sentence_img_aug_cc3m_a2_sc05/checkpoints/epoch_30.pt
# CHKPNT=/lpai/models/sentence-clip/itocc3m/sentence_img_aug_cc3m_a2_sc05/checkpoints/epoch_30.pt
# CHKPNT=/lpai/models/ito/dm3mtxt2/ITO_reca_sub2_lr1e3_bs2048/checkpoints/epoch_30.pt
# CHKPNT=/lpai/volumes/jfs-data-lhp-bd-ga/ito_dreamcc3m/epoch_30.pt
# CHKPNT=/lpai/volumes/jfs-data-lhp-bd-ga/ito_recap/ITO_reca_sub3_lr1e3_bs2048/checkpoints/epoch_30.pt
# CHKPNT=/lpai/volumes/jfs-data-lhp-bd-ga/ito_cc3m/sub2lr3e3bs4k/epoch_30.pt
CHKPNT=/lpai/volumes/jfs-data-lhp-bd-ga/ito-cc12m/sub2lr1e3b2k/epoch_30.pt
# CHKPNT=/lpai/volumes/jfs-data-lhp-bd-ga/ito-cc12m/sub2lr1e3b4k/epoch_30.pt

SKETCH=/lpai/volumes/so-volume-bd-ga/lhp/datasets/imagenetsketch/data/sketch
# V2=path/to/datasets/imagenetv2-matched-frequency-format-val
A=/lpai/volumes/so-volume-bd-ga/lhp/datasets/imagenet-a/test_data
# O=path/to/datasets/imagenet-o
R=/lpai/volumes/so-volume-bd-ga/lhp/datasets/imagenet-r/test_data

CUDA_VISIBLE_DEVICES=5 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-sketch=$SKETCH
# CUDA_VISIBLE_DEVICES=4 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-a=$A
# python -m main.run --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-o=$O
# CUDA_VISIBLE_DEVICES=4 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-r=$R
# python -m main.run --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-v2=$V2