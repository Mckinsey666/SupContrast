for e in 200
do
python3 main_supcon.py --batch_size 1024 \
  --learning_rate 0.5 \
  --temp 0.5 \
  --cosine \
  --method SimCLR \
  --model resnet18 \
  --epochs 1000 \
  --use_learned_aug \
  --dataset cifar100 \
  --policy jigsaw_cifar100_epoch$e \
  --trial 825
done