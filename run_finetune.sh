for e in ckpt_epoch_200 ckpt_epoch_400 ckpt_epoch_600 
do
python3 main_finetune.py --batch_size 512 \
  --learning_rate 1 \
  --model resnet18 \
  --epochs 100 \
  --dataset cifar100 \
  --ckpt save/SupCon/cifar100_models/SimCLR_cifar100_resnet18_lr_0.5_decay_0.0001_bsz_1024_temp_0.5_trial_825_cosine_augpolicy_jigsaw_cifar100_epoch299_warm/$e\.pth
done
