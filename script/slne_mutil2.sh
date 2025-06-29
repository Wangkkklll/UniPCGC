CUDA_VISIBLE_DEVICES=1 python train_lossy.py \
    --prefix sopa_one_slne_slim \
    --batch_size 1 \
    --lr 8e-4 \
    --epoch 1 \
    --check_time 1\
    --scale 8 \
    --alpha 2.0 \
    --beta 2.0 \
    --entropy_model 'f' \
    --muti_lambda 1.0 \
    --channel 32 \
    --stage stage1 

CUDA_VISIBLE_DEVICES=1 python train_lossy.py \
    --prefix sopa_one_slne_slim \
    --batch_size 1 \
    --lr 8e-4 \
    --epoch 2 \
    --check_time 1\
    --scale 8 \
    --alpha 2.0 \
    --beta 2.0 \
    --entropy_model 'f' \
    --muti_lambda 1.0 \
    --channel 32 \
    --stage stage2 \
    --pre_loading ./ckpts/sopa_one_slne_slim/epoch_0.pth


CUDA_VISIBLE_DEVICES=1 python train_lossy.py \
    --prefix sopa_one_slne_slim \
    --batch_size 1 \
    --lr 8e-4 \
    --epoch 25 \
    --check_time 1\
    --scale 8 \
    --alpha 2.0 \
    --beta 2.0 \
    --entropy_model 'f' \
    --muti_lambda 0.3 3.0 \
    --channel 32 \
    --stage stage3 \
    --pre_loading ./ckpts/sopa_one_slne_slim/epoch_1.pth