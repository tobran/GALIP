data=coco
cfg=./cfg/model/${data}.yml

train=True
multi_gpus=True

batch_size=64
z_dim=100

state_epoch=1 
pretrained_model_path='./saved_models/coco/model_save_file'
log_dir='new'

npz_path='./code/text2img/metrics/FID/coco_val256_g0.npz'

nodes=8
num_workers=8
master_port=11266
mixed_precision=True
stamp=gpu${nodes}MP_${mixed_precision}

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=$nodes --master_port=$master_port src/train.py \
                    --stamp $stamp \
                    --cfg $cfg \
                    --mixed_precision $mixed_precision \
                    --z_dim $z_dim \
                    --log_dir $log_dir \
                    --batch_size $batch_size \
                    --state_epoch $state_epoch \
                    --GPUs $nodes \
                    --npz_path $npz_path \
                    --num_workers $num_workers \
                    --train $train \
                    --multi_gpus $multi_gpus \
                    --pretrained_model_path $pretrained_model_path \
