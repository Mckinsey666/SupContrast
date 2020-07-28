<<<<<<< HEAD
for e in 0
do
python3 main_linear.py --batch_size 512 \
  --learning_rate 1 \
  --model resnet18 \
  --epochs 100 \
  --ckpt save/SupCon/cifar10_models/SimCLR_cifar10_resnet18_lr_0.5_decay_0.0001_bsz_1024_temp_0.5_trial_0_cosine_augpolicy_contrast_rotate_cifar10_epoch$e\_resized_crop_warm/ckpt_epoch_200.pth
=======
for e in 20 50 100 150 199
do
  echo "Running epoch $e"
    python3 main_linear.py --batch_size 512 \
  --learning_rate 1 \
  --model resnet18 \
  --epochs 100 \
  --ckpt save/SupCon/cifar10_models/SimCLR_cifar10_resnet18_lr_0.5_decay_0.0001_bsz_1024_temp_0.5_trial_0_cosine_augpolicy_contrast_rotate_cifar10_v2_epoch$e\_all_resized_crop_warm/ckpt_epoch_200.pth
>>>>>>> 84946bf8e7c8867df189a43e80b2410b3ef78a05
done