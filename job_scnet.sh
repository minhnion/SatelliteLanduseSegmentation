BATCH_SIZE=64
LR=0.0001
EPOCH=100
DATASET=deepglobe

python main.py --cuda --model=SCNet --gpu_id=0 --lr=$LR --epoch=$EPOCH --batch_size=$BATCH_SIZE --dataset=$DATASET --pretrained=$PRETRAINED
echo 'Done'
