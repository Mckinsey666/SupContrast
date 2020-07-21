for p in 10 20 50 75 100 120 150
do
python3 main_supcon.py --batch_size 1024 \
  --learning_rate 0.5 \
  --temp 0.5 \
  --cosine \
  --method SimCLR \
  --model resnet18 \
  --epochs 200 \
  --use_learned_aug \
  --policy contrastive_no_pretext_reduce_cifar10_epoch$p\_top25
done
