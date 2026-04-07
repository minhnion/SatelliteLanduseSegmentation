BATCH_SIZE=2
python main.py --cuda --model=PretrainedViTUNet --gpu_id=0 --lr=0.0001 --epoch=200 --depth=2 --heads=16 --dropout=0.2 --batch_size=$BATCH_SIZE
