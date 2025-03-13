BATCH_SIZE=32
LR=0.0001
EPOCH=30

python main.py --cuda --model=CrossAttentionUNetSR --gpu_id=0 --lr=$LR --epoch=$EPOCH --batch_size=$BATCH_SIZE
