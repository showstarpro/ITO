# !/bin/bash

OUTPATH=/path/to/output/models/downstreams/retrieval
BS=128
MODEL="ViT-B-16"
#MODEL="ViT-B-16-512"
#MODEL="ViT-L-16"

CHKPNT=/path/to/models/checkpoints/epoch_30.pt

MSCOCO=/path/to/datasets/coco2017/val2017
MSCOCO_ANNOT=/path/to/datasets/coco2017/annotations/captions_val2017.json
FLICKR=/path/to/datasets/flickr30k/flickr30k-images/
FLICKR_ANNOT=/path/to/datasets/flickr30k/flickr30k_val.json
DOCCI=/path/to/docci-test.arrow

# CUDA_VISIBLE_DEVICES=4 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --ms-coco=$MSCOCO --ms-coco-annot=$MSCOCO_ANNOT
# CUDA_VISIBLE_DEVICES=4 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --flickr=$FLICKR --flickr-annot=$FLICKR_ANNOT
CUDA_VISIBLE_DEVICES=4 python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --docci=$DOCCI