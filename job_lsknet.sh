BATCH_SIZE=32 # 2GB
LR=0.00005
EPOCH=100
DATASET=deepglobe
PRETRAINED=/mnt/hungvv/minh/log/weights/dataset:deepglobe-classification/model:FoundationModel/datetime:20250325_042112_epoch:5_bs:4_lr:5e-05/weight.pth

python main.py --cuda --model=LSKNet --gpu_id=0 --lr=$LR --epoch=$EPOCH --batch_size=$BATCH_SIZE --dataset=$DATASET --pretrained=$PRETRAINED
echo 'Done'
