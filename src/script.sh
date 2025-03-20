source /root/anaconda3/etc/profile.d/conda.sh 

conda activate superclass

export WANDB_API_KEY=7352f9a349b74e01062672d0bc0bd3a8094677e2

# python --nproc_per_node 8 -m open_clip_train.main \
#     --save-frequency 1 \
#     --zeroshot-frequency 1 \
#     --report-to tensorboard \
#     --train-data="/lpai/dataset/datacomp-13m/0-1-0/datacomp_small/shards/{00000000..00001287}.tar"  \
#     --val-data="/lpai/dataset/imagenet-1k/0-1-0/ILSVRC2012/val"  \
#     --dataset_type webdataset \
#     --imagenet-val="/lpai/dataset/imagenet-1k/0-1-0/ILSVRC2012/val" \
#     --train-num-samples 512_000_000 \
#     --warmup 10000 \
#     --batch-size=4096 \
#     --lr=1e-3 \
#     --wd=0.1 \
#     --epochs=30 \
#     --workers=8 \
#     --model ViT-B-16

# torchrun --nproc_per_node 8 -m open_clip_train.main \
#     --train-data "/lpai/dataset/datacomp-13m/0-1-0/datacomp_small/shards/{00000000..00001287}.tar" \
#     --train-num-samples 10968539 \
#     --dataset-type webdataset \
#     --batch-size 512 \
#     --precision amp_bfloat16 \
#     --workers 8 \
#     --imagenet-val "/lpai/dataset/imagenet-1k/0-1-0/ILSVRC2012/val" \
#     --model ViT-B-16 \
#     --name 'vit-b-clip' \
#     --report-to "wandb" \
#     --wandb-project-name "vit-b-clip" 

# export CUDA_VISIBLE_DEVICES=6,7



torchrun --nproc_per_node 8 -m open_clip_train.main \
    --train-data "/lpai/dataset/cc12m/0-1-0/cc12m-wds/cc12m-train-{0000..2175}.tar"  \
    --train-num-samples 10968539 \
    --dataset-type webdataset \
    --batch-size 512 \
    --precision amp \
    --workers 16 \
    --lr 1e-3 --wd 0.1 --warmup 10000  --beta1 0.9 --beta2 0.98 --eps 1e-06 \
    --epochs 30 \
    --model "ViT-B-16" \
    --report-to wandb  --wandb-project-name sentence \
    --coca-caption-loss-weight 0 --coca-contrastive-loss-weight 0 \
    --name sentence_2 --alpha 2


# torchrun --nproc_per_node 1 -m open_clip_train.main \
#     --save-frequency 1 \
#     --save-most-recent \
#     --zeroshot-frequency 1 \
#     --train-data "/lpai/dataset/datacomp-13m/0-1-0/datacomp_small/shards/{00000000..00001287}.tar" \
#     --dataset-type webdataset \
#     --lr "2.048e-3" \
#     --beta1 0.9 \
#     --beta2 0.95 \
#     --warmup 782 \
#     --wd 0.2 \
#     --batch-size 4096 \
#     --aug-cfg scale='(0.4, 1.0)' color_jitter='(0.32, 0.32, 0.32, 0.08)' color_jitter_prob=0.8 gray_scale_prob=0.2 \
#     --epochs=7 \
#     --workers=6 \
#     --model ViT-B-16 \
#     --precision 'amp_bf16' \
#     --gather-with-grad \
#     --force-image-size 84 \
#     --grad-checkpointing \
#     --log-every-n-steps 32 \
#     --seed 0 \
#     --log-dir /lpai/volumes/so-volume-ga/lhp/clip/logs/ \
#     --imagenet-val "/lpai/dataset/imagenet-1k/0-1-0/ILSVRC2012/val" \
#     --name 'vit-b-clip' \
#     --report-to "wandb" \
#     --wandb-project-name "vit-b-clip"