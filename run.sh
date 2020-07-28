for e in 50 100 200 300 400
do
python3 main_supcon.py --batch_size 1024 \
  --learning_rate 0.5 \
  --temp 0.5 \
  --cosine \
  --method SimCLR \
  --model resnet18 \
  --epochs 200 \
  --use_learned_aug \
  --policy contrast_rotate_cifar10_v1_epoch$e\_all \
  --use_resized_crop
done
