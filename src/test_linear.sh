#!/bin/bash

OUTPATH=/path/to/output/models/downstreams/linear_probe
BS=128
MODEL="ViT-B-16"
# #MODEL="ViT-B-16-512"
# #MODEL="ViT-L-16"

CHKPNT=/path/to/models/checkpoints/epoch_30.pt

CIFAR10=/path/to/datasets/cifar10
CIFAR100=/path/to/datasets/cifar100
IMAGENETVAL=/path/to/datasets/imagenet-1k/val
IMAGENETTRAIN=/path/to/datasets/imagenet-1k/train
FLOWERS102=/path/to/datasets/flower_102
FOOD101=/path/to/datasets/food_101
PETS=/path/to/datasets/pets
STANFORD=/path/to/datasets/stanford_cars

python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --cifar10=$CIFAR10 --test-linear
# python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --cifar100=$CIFAR100 --test-linear
# python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-val=$IMAGENETVAL --imagenet-train=$IMAGENETTRAIN --test-linear
# python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --flowers-102=$FLOWERS102 --test-linear
# python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --food-101=$FOOD101 --test-linear
# python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --pets=$PETS --test-linear
# python -m open_clip_train.main --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --stanford=$STANFORD --test-linear