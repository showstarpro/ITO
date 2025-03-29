#!/bin/bash

OUTPATH=/lpai/output/models/openclipcc12m/downstreams/retrieval-flickr30k
BS=16
MODEL="ViT-B-16"
#MODEL="ViT-B-16-512"
#MODEL="ViT-L-16"

CHKPNT=/lpai/models/openclip/vitblr1e3/open_clip/clip-vitb16-cc12m-epochs30/checkpoints/epoch_30.pt
# CHKPNT=/lpai/models/openclip/openclipcc3m/open_clip_cc3m_1/checkpoints/epoch_30.pt

# MSCOCO=/lpai/dataset/coco2017/0-1-0/coco/val2017
# MSCOCO_ANNOT=/lpai/dataset/coco2017/0-1-0/coco/annotations/captions_val2017.json
FLICKR=/lpai/volumes/so-volume-bd-ga/lhp/datasets/flickr30k/flickr30k-images/
FLICKR_ANNOT=/lpai/volumes/so-volume-bd-ga/lhp/datasets/flickr30k/flickr30k_annotations_test.json

# CUDA_VISIBLE_DEVICES=4 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --ms-coco=$MSCOCO --ms-coco-annot=$MSCOCO_ANNOT
CUDA_VISIBLE_DEVICES=4 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --flickr=$FLICKR --flickr-annot=$FLICKR_ANNOT