python3 main_supcon.py --batch_size 512 \
  --accum_grad 2 \
  --learning_rate 0.5 \
  --temp 0.5 \
  --cosine \
  --method SimCLR \
  --model resnet18 \
  --epochs 500
