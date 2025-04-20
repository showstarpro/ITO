source /root/anaconda3/etc/profile.d/conda.sh 

conda activate superclass

export WANDB_API_KEY=7352f9a349b74e01062672d0bc0bd3a8094677e2
# export CUDA_VISIBLE_DEVICES=6,7


torchrun --nproc_per_node 8 -m open_clip_train.main \
    --train-data "/lpai/dataset/cc12m/0-1-0/cc12m-wds/cc12m-train-{0000..2175}.tar"  \
    --train-num-samples 10968539 \
    --dataset-type webdataset \
    --batch-size 256 \
    --precision amp \
    --workers 16 \
    --lr 1e-3 --wd 0.1 --warmup 10000  --beta1 0.9 --beta2 0.98 --eps 1e-06 \
    --epochs 30 \
    --model "ViT-B-16" \
    --coca-caption-loss-weight 0 --coca-contrastive-loss-weight 0 \
    --name sentence_img_aug_a2_test1 --alpha 2 \
    --aug \
    --aug-cfg gray_scale_prob=0.2 color_jitter=0.4,0.4,0.4,0.1 color_jitter_prob=0.8 scale=0.5,1.0 
