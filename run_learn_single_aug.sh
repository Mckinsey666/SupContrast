python3 main_single_aug_pretext.py --batch_size 1024 \
    --learning_rate 0.5 \
    --temp 0.5 \
    --cosine \
    --method SimCLR \
    --model resnet18 \
    --epochs 200 \
    --aug_type crop \
    --aug_levels 5