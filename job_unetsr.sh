BATCH_SIZE=4
LR=0.0001
EPOCH=40

python main.py --cuda --model=UNetSR --gpu_id=0 --lr=$LR --epoch=$EPOCH --batch_size=$BATCH_SIZE
