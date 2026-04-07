BATCH_SIZE=2
python test.py --cuda --model=UNetSR --gpu_id=0 --lr=0.0001 --epoch=5 --batch_size=$BATCH_SIZE
