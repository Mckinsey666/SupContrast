for policy in "contrast_cifar10_epoch20_top25" "contrast_cifar10_epoch50_top25" "contrast_cifar10_epoch100_top25"
do
python3 main_supcon.py --batch_size 1024 \
  --learning_rate 0.5 \
  --temp 0.5 \
  --cosine \
  --method SimCLR \
  --model resnet18 \
  --epochs 200 \
  --use_learned_aug \
  --policy $policy
done
