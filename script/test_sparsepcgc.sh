CUDA_VISIBLE_DEVICES=3 python train_lossy.py \
    --prefix one_sopa_slne_test \
    --batch_size 4 \
    --lr 8e-4 \
    --epoch 2 \
    --check_time 1\
    --scale 6 \
    --alpha 1.0 \
    --beta 0.0 \
    --entropy_model 'f' \
    --muti_lambda 1.0 \
    --channel 32 \
    --stage stage2

CUDA_VISIBLE_DEVICES=3 python train_lossy.py \
    --prefix one_sopa_slne_test \
    --batch_size 4 \
    --lr 8e-4 \
    --epoch 2 \
    --check_time 1\
    --scale 6 \
    --alpha 1.0 \
    --beta 0.2 \
    --entropy_model 'f' \
    --muti_lambda 1.0 \
    --channel 32 \
    --pre_loading ./ckpts/one_sopa_slne_test/epoch_0.pth \
    --stage stage2

CUDA_VISIBLE_DEVICES=3 python train_lossy.py \
    --prefix one_sopa_slne_test \
    --batch_size 4 \
    --lr 8e-4 \
    --epoch 25 \
    --check_time 1\
    --scale 6 \
    --alpha 1.0 \
    --beta 1.0 \
    --entropy_model 'f' \
    --muti_lambda 1.0 \
    --channel 32 \
    --pre_loading ./ckpts/one_sopa_slne_test/epoch_0.pth \
    --stage stage2