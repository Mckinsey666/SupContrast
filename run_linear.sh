touch 'aug_crop_comp.txt'
for s in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
  echo "Running strength $s"
    python3 main_linear.py --batch_size 512 \
  --learning_rate 1 \
  --model resnet18 \
  --epochs 100 \
  --ckpt ~/brian/SupContrast/save/SupCon/cifar10_models/SimCLR_cifar10_resnet18_lr_0.5_decay_0.0001_bsz_1024_temp_0.5_trial_comp_aug_crop_$s\_cosine_warm/ckpt_epoch_200.pth
done