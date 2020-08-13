for e in 999
do
python3 main_linear.py --batch_size 512 \
  --learning_rate 1 \
  --model resnet18 \
  --epochs 100 \
  --ckpt save/SupCon/cifar10_models/SimCLR_cifar10_resnet18_lr_0.5_decay_0.0001_bsz_1024_temp_0.5_trial_raw_cosine_augpolicy_default_jigsaw_epoch$e\_warm/ckpt_epoch_200.pth
done
