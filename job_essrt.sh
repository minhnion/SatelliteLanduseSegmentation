BATCH_SIZE=16
LR=0.00005
EPOCH=60
DATASET=dubai

python main.py --cuda --model=ESSRT --gpu_id=0 --lr=$LR --epoch=$EPOCH --batch_size=$BATCH_SIZE --dataset=$DATASET
