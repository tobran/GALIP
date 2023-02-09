cfg=$1
batch_size=64

pretrained_model='./saved_models/data/model_save_file/xxx.pth'
multi_gpus=True
mixed_precision=True

nodes=1
num_workers=8
master_port=11277
stamp=gpu${nodes}MP_${mixed_precision}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$nodes --master_port=$master_port src/test.py \
                    --stamp $stamp \
                    --cfg $cfg \
                    --mixed_precision $mixed_precision \
                    --batch_size $batch_size \
                    --num_workers $num_workers \
                    --multi_gpus $multi_gpus \
                    --pretrained_model_path $pretrained_model \
