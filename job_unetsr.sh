BATCH_SIZE=16
LR=0.0001
EPOCH=10
DATASET=north_vn
PRETRAINED=model/UNetSR/best_sr_first.pth
python main.py --cuda --model=UNetSR --gpu_id=0 --lr=$LR --epoch=$EPOCH --batch_size=$BATCH_SIZE --pretrained=$PRETRAINED --dataset=$DATASET