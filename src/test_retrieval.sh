#!/bin/bash

# OUTPATH=/lpai/output/models/ito-cc12m-sub2lr1e3b2k/downstreams/retrieval-coco
# BS=128
# MODEL="ViT-B-16"
# #MODEL="ViT-B-16-512"
# #MODEL="ViT-L-16"

# # CHKPNT=/lpai/models/sentence-clip/itocc12m/sentence_img_aug_cc3m_a2_sc05/checkpoints/epoch_30.pt
# # CHKPNT=/lpai/models/sentence-clip/itocc3m/sentence_img_aug_cc3m_a2_sc05/checkpoints/epoch_30.pt
# # CHKPNT=/lpai/models/ito/dm3mtxt2/ITO_reca_sub2_lr1e3_bs2048/checkpoints/epoch_30.pt
# # CHKPNT=/lpai/volumes/jfs-data-lhp-bd-ga/ito_dreamcc3m/epoch_30.pt
# # CHKPNT=/lpai/volumes/jfs-data-lhp-bd-ga/ito_recap/ITO_reca_sub3_lr1e3_bs2048/checkpoints/epoch_30.pt
# # CHKPNT=/lpai/volumes/jfs-data-lhp-bd-ga/ito_cc3m/sub2lr3e3bs4k/epoch_30.pt
# CHKPNT=/lpai/volumes/jfs-data-lhp-bd-ga/ito-cc12m/sub2lr1e3b2k/epoch_30.pt
# # CHKPNT=/lpai/volumes/jfs-data-lhp-bd-ga/ito-cc12m/sub2lr1e3b4k/epoch_30.pt

# MSCOCO=/lpai/dataset/coco2017/0-1-0/coco/val2017
# MSCOCO_ANNOT=/lpai/dataset/coco2017/0-1-0/coco/annotations/captions_val2017.json
# FLICKR=/lpai/volumes/so-volume-bd-ga/lhp/datasets/flickr30k/flickr30k-images/
# FLICKR_ANNOT=/lpai/volumes/so-volume-bd-ga/lhp/datasets/flickr30k/flickr30k_val.json
# DOCCI=/lpai/volumes/jfs-data-lhp-bd-ga/google___docci/docci/1.0.0/d38d507c9795602d4e50dcd02e1e1dc4fa8c58ac69c2949c5c3a20c9d00c5b8b/docci-test.arrow

# CUDA_VISIBLE_DEVICES=4 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --ms-coco=$MSCOCO --ms-coco-annot=$MSCOCO_ANNOT
# # CUDA_VISIBLE_DEVICES=4 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --flickr=$FLICKR --flickr-annot=$FLICKR_ANNOT
# # CUDA_VISIBLE_DEVICES=4 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --docci=$DOCCI



#!/bin/bash

OUTPATH=/lpai/output/models/dm3mtxt2-ITO_reca_sub2_lr1e3_bs2048/downstreams/linear_probe-img
BS=128
MODEL="ViT-B-16"
#MODEL="ViT-B-16-512"
#MODEL="ViT-L-16"

# CHKPNT=/lpai/models/sentence-clip/itocc12m/sentence_img_aug_cc3m_a2_sc05/checkpoints/epoch_30.pt
CHKPNT=/lpai/models/ito/dm3mtxt2/ITO_reca_sub2_lr1e3_bs2048/checkpoints/epoch_30.pt
# CHKPNT=/lpai/volumes/jfs-data-lhp-bd-ga/ito_dreamcc3m/epoch_30.pt
# CHKPNT=/lpai/volumes/jfs-data-lhp-bd-ga/ito_recap/ITO_reca_sub3_lr1e3_bs2048/checkpoints/epoch_30.pt
# CHKPNT=/lpai/volumes/jfs-data-lhp-bd-ga/ito_cc3m/sub2lr3e3bs4k/epoch_30.pt
# CHKPNT=/lpai/volumes/jfs-data-lhp-bd-ga/ito-cc12m/sub2lr1e3b2k/epoch_30.pt
# CHKPNT=/lpai/volumes/jfs-data-lhp-bd-ga/ito-cc12m/sub2lr1e3b4k/epoch_30.pt

CIFAR10=/lpai/volumes/so-volume-bd-ga/lhp/datasets/cifar10
CIFAR100=/lpai/volumes/so-volume-bd-ga/lhp/datasets/cifar100
IMAGENETVAL=/lpai/dataset/imagenet-1k/0-1-0/ILSVRC2012/val
IMAGENETTRAIN=/lpai/dataset/imagenet-1k/0-1-0/train
FLOWERS102=/lpai/volumes/so-volume-bd-ga/lhp/datasets/flower_102/dataset
FOOD101=/lpai/volumes/so-volume-bd-ga/lhp/datasets/food_101
PETS=/lpai/volumes/so-volume-bd-ga/lhp/datasets/pets
STANFORD=/lpai/volumes/so-volume-bd-ga/lhp/datasets

# python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-val=$IMAGENETVAL
# CUDA_VISIBLE_DEVICES=7 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --cifar10=$CIFAR10 --test-linear
# CUDA_VISIBLE_DEVICES=4 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --cifar100=$CIFAR100 --test-linear
CUDA_VISIBLE_DEVICES=4 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-val=$IMAGENETVAL --imagenet-train=$IMAGENETTRAIN --test-linear
# CUDA_VISIBLE_DEVICES=4 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --flowers-102=$FLOWERS102 --test-linear
# CUDA_VISIBLE_DEVICES=3 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --food-101=$FOOD101 --test-linear
# CUDA_VISIBLE_DEVICES=4 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --pets=$PETS --test-linear
# CUDA_VISIBLE_DEVICES=4 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --stanford=$STANFORD --test-linear
