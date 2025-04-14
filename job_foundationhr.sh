BATCH_SIZE=1
LR=0.00005
EPOCH=100
DATASET=sentinelhr
# PRETRAINED=/mnt/hungvv/minh/log/weights/dataset:sentinel_hr_lr_dataset_cut/model:FoundationKDModelHR/datetime:20250410_210242_epoch:100_bs:8_lr:5e-05/weight.pth

python main.py --cuda --model=FoundationKDModelHR --gpu_id=0 --lr=$LR --epoch=$EPOCH --batch_size=$BATCH_SIZE --dataset=$DATASET --pretrained=$PRETRAINED
echo 'Done'
