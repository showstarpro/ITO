source /root/anaconda3/etc/profile.d/conda.sh 

conda activate superclass

export WANDB_API_KEY=7352f9a349b74e01062672d0bc0bd3a8094677e2
# export CUDA_VISIBLE_DEVICES=6,7


torchrun --nproc_per_node 8 -m open_clip_train.main \
    --train-data '/lpai/dataset/cc3m-3long-3short-1raw/0-1-0/cc3m_3long_3short_1raw_captions/00{000..287}.tar' \
    --train-num-samples  1711097 \
    --dataset-type webdataset \
    --batch-size 256 \
    --precision amp \
    --workers 16 \
    --lr 1e-3 --wd 0.1 --warmup 10000  --beta1 0.9 --beta2 0.98 --eps 1e-06 \
    --epochs 30 \
    --model "ViT-B-16" \
    --coca-caption-loss-weight 0 --coca-contrastive-loss-weight 0 \
    --name ito_text_tt --alpha 2 \
    --aug \
    --aug-cfg gray_scale_prob=0.2 color_jitter=0.4,0.4,0.4,0.1 color_jitter_prob=0.8 scale=0.5,1.0 
