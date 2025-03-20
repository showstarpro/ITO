# python -m open_clip_train.main \
#     --imagenet-val="/lpai/dataset/imagenet-1k/0-1-0/ILSVRC2012/val"  \
#     --model RN50 \
#     --pretrained "/lpai/open_clip-main/src/checkpoints/rn50-quickgelu-cc12m-f000538c.pt"
    # --pretrained "/lpai/open_clip-main/src/logs/vit-b-clip/checkpoints/epoch_32.pt"
CUDA_VISIBLE_DEVICES=2 python -m open_clip_train.main \
    --imagenet-val="/lpai/dataset/imagenet-1k/0-1-0/ILSVRC2012/val"  \
    --model ViT-B-16 \
    --pretrained "/lpai/models/openclip/shareclip1e3/logs/share_clip_ep30_cc12m/checkpoints/epoch_30.pt"