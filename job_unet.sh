BATCH_SIZE=2
python main.py --cuda --model=ViTUnet --gpu_id=0 --lr=0.00001 --epoch=10 --batch_size=$BATCH_SIZE
