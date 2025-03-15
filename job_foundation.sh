BATCH_SIZE=4
LR=0.00005
EPOCH=100

python main.py --cuda --model=FoundationModel --gpu_id=0 --lr=$LR --epoch=$EPOCH --batch_size=$BATCH_SIZE
