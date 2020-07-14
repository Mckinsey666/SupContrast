for s in 0.1 0.2 0.3 0.4 0.5
do
python3 main_ce.py --batch_size 1024 \
  --learning_rate 0.8 \
  --cosine \
  --model resnet18 \
  --epochs 300 \
  --strength $s \
  --trial ce_crop$s
done
