<<<<<<< HEAD
for e in 0
=======
for e in 20 50 100 150 199
>>>>>>> 84946bf8e7c8867df189a43e80b2410b3ef78a05
do
python3 main_supcon.py --batch_size 1024 \
  --learning_rate 0.5 \
  --temp 0.5 \
  --cosine \
  --method SimCLR \
  --model resnet18 \
  --epochs 200 \
  --use_learned_aug \
<<<<<<< HEAD
  --policy contrast_rotate_cifar10_epoch$e \
=======
  --policy contrast_rotate_cifar10_v2_epoch$e\_all \
>>>>>>> 84946bf8e7c8867df189a43e80b2410b3ef78a05
  --use_resized_crop
done