CUDA_VISIBLE_DEVICES=4 python train_lossy.py \
    --prefix sopa_one_slne_mutil \
    --batch_size 4 \
    --lr 8e-4 \
    --epoch 1 \
    --check_time 1\
    --scale 7 \
    --alpha 1.0 \
    --beta 1.0 \
    --entropy_model 'f' \
    --muti_lambda 1.0 \
    --channel 32 \
    --stage stage1


CUDA_VISIBLE_DEVICES=4 python train_lossy.py \
    --prefix sopa_one_slne_mutil \
    --batch_size 4 \
    --lr 8e-4 \
    --epoch 2 \
    --check_time 1\
    --scale 7 \
    --alpha 1.0 \
    --beta 1.0 \
    --entropy_model 'f' \
    --muti_lambda 1.0 \
    --channel 32 \
    --stage stage2 \
    --pre_loading ./ckpts/sopa_one_slne_mutil/epoch_0.pth

CUDA_VISIBLE_DEVICES=4 python train_lossy.py \
    --prefix sopa_one_slne_mutil \
    --batch_size 4 \
    --lr 8e-4 \
    --epoch 25 \
    --check_time 1\
    --scale 7 \
    --alpha 1.0 \
    --beta 1.0 \
    --entropy_model 'f' \
    --muti_lambda 0.3 3.0 \
    --channel 32 \
    --stage stage3 \
    --pre_loading ./ckpts/sopa_one_slne_mutil/epoch_1.pth

