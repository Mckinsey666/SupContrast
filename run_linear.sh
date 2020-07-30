<<<<<<< HEAD
for e in 0
=======
for e in 200 300 400
>>>>>>> effdb893cb942f26fb7d3eeb1e4e7b656b68e629
do
python3 main_linear.py --batch_size 512 \
  --learning_rate 1 \
  --model resnet18 \
  --epochs 100 \
<<<<<<< HEAD
  --ckpt save/SupCon/cifar10_models/SimCLR_cifar10_resnet18_lr_0.5_decay_0.0001_bsz_1024_temp_0.5_trial_0_cosine_augpolicy_contrast_rotate_cifar10_epoch$e\_resized_crop_warm/ckpt_epoch_200.pth
done
=======
  --ckpt save/SupCon/cifar10_models/SimCLR_cifar10_resnet18_lr_0.5_decay_0.0001_bsz_1024_temp_0.5_trial_0_cosine_augpolicy_cifar10_default_set_epoch$e\_resized_crop_warm/ckpt_epoch_200.pth
done
>>>>>>> effdb893cb942f26fb7d3eeb1e4e7b656b68e629
