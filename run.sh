for e in 999
do
python3 main_supcon.py --batch_size 1024 \
  --learning_rate 0.5 \
  --temp 0.5 \
  --cosine \
  --method SimCLR \
  --model resnet18 \
  --epochs 200 \
  --use_learned_aug \
  --policy default_jigsaw_epoch$e \
  --trial raw
done