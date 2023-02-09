cfg=$1
batch_size=64

state_epoch=1 
pretrained_model_path='./saved_models/data/model_save_file'
log_dir='new'

multi_gpus=True
mixed_precision=True

nodes=8
num_workers=8
master_port=11266
stamp=gpu${nodes}MP_${mixed_precision}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=$nodes --master_port=$master_port src/train.py \
                    --stamp $stamp \
                    --cfg $cfg \
                    --mixed_precision $mixed_precision \
                    --log_dir $log_dir \
                    --batch_size $batch_size \
                    --state_epoch $state_epoch \
                    --num_workers $num_workers \
                    --multi_gpus $multi_gpus \
                    --pretrained_model_path $pretrained_model_path \
