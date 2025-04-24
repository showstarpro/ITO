#!/bin/bash

OUTPATH=/lpai/output/models/ITO_reca_sub2_lr1e3_bs2048/downstreams/retrieval-flickr
BS=128
MODEL="ViT-B-16"
#MODEL="ViT-B-16-512"
#MODEL="ViT-L-16"

# CHKPNT=/lpai/models/sentence-clip/itocc12m/sentence_img_aug_cc3m_a2_sc05/checkpoints/epoch_30.pt
# CHKPNT=/lpai/models/sentence-clip/itocc3m/sentence_img_aug_cc3m_a2_sc05/checkpoints/epoch_30.pt
CHKPNT=/lpai/models/ito/dm3mtxt2/ITO_reca_sub2_lr1e3_bs2048/checkpoints/epoch_30.pt

# MSCOCO=/lpai/dataset/coco2017/0-1-0/coco/val2017
# MSCOCO_ANNOT=/lpai/dataset/coco2017/0-1-0/coco/annotations/captions_val2017.json
FLICKR=/lpai/volumes/so-volume-bd-ga/lhp/datasets/flickr30k/flickr30k-images/
FLICKR_ANNOT=/lpai/volumes/so-volume-bd-ga/lhp/datasets/flickr30k/flickr30k_val.json
# DOCCI=/lpai/volumes/jfs-data-lhp-bd-ga/google___docci/docci/1.0.0/d38d507c9795602d4e50dcd02e1e1dc4fa8c58ac69c2949c5c3a20c9d00c5b8b/docci-test.arrow

# CUDA_VISIBLE_DEVICES=7 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --ms-coco=$MSCOCO --ms-coco-annot=$MSCOCO_ANNOT
CUDA_VISIBLE_DEVICES=7 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --flickr=$FLICKR --flickr-annot=$FLICKR_ANNOT
# CUDA_VISIBLE_DEVICES=7 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --docci=$DOCCI