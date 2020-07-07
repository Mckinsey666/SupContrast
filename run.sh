for s in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do 
  python3 main_single_aug.py --batch_size 1024 \
    --learning_rate 0.5 \
    --temp 0.5 \
    --cosine \
    --method SimCLR \
    --model resnet18 \
    --epochs 200 \
    --aug_type crop \
    --aug_strength $s \
    --trial comp
done