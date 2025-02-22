BATCH_SIZE=32
python infer.py --cuda --model=ViTUnet --gpu_id=0 --lr=0.0001 --epoch=200 --batch_size=$BATCH_SIZE
