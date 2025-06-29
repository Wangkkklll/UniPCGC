CUDA_VISIBLE_DEVICES=1 python train_lossy.py \
    --prefix one_sopa \
    --batch_size 8 \
    --lr 8e-4 \
    --epoch 25 \
    --check_time 1\
    --scale 5 \
    --alpha 1.0 \
    --entropy_model 'f' \
    --muti_lambda 1.0 \
    --channel 32 \



