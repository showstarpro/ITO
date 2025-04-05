#!/bin/bash

OUTPATH=/lpai/output/models/siglip12m1e3/downstreams/retrieval-coco
BS=16
MODEL="ViT-B-16-SigLIP"
#MODEL="ViT-B-16-512"
#MODEL="ViT-L-16"

# CHKPNT=/lpai/models/sentence-clip/siglip3m1e3/siglip3m_1e3/checkpoints/epoch_30.pt
CHKPNT=/lpai/models/sentence-clip/siglip12m1e3/siglip12m_1e3/checkpoints/epoch_30.pt

MSCOCO=/lpai/dataset/coco2017/0-1-0/coco/val2017
MSCOCO_ANNOT=/lpai/dataset/coco2017/0-1-0/coco/annotations/captions_val2017.json
# FLICKR=/lpai/volumes/so-volume-bd-ga/lhp/datasets/flickr30k/flickr30k-images/
# FLICKR_ANNOT=/lpai/volumes/so-volume-bd-ga/lhp/datasets/flickr30k/flickr30k_annotations_test.json

CUDA_VISIBLE_DEVICES=4 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --ms-coco=$MSCOCO --ms-coco-annot=$MSCOCO_ANNOT
# CUDA_VISIBLE_DEVICES=4 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --flickr=$FLICKR --flickr-annot=$FLICKR_ANNOT