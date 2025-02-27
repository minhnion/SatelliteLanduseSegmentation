BATCH_SIZE=16
LR=0.00005
EPOCH=10
PRETRAINED=/mnt/anhtn/log/weights/dataset:new_13bands_dataset_splitted/model:UNetSR/epoch:40_bs:4_lr:0.0001_datetime:20250222_214304/best_weight.pth

python main.py --cuda --model=UNetSR --gpu_id=0 --lr=$LR --epoch=$EPOCH --batch_size=$BATCH_SIZE --pretrained=$PRETRAINED
