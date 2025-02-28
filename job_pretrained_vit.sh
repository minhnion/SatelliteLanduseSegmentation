BATCH_SIZE=8
python main.py --cuda --model=PretrainedViT --gpu_id=0 --lr=0.0001 --epoch=200 --batch_size=$BATCH_SIZE
python main.py --cuda --model=PretrainedViT --gpu_id=0 --lr=0.00005 --epoch=200 --batch_size=$BATCH_SIZE
python main.py --cuda --model=PretrainedViT --gpu_id=0 --lr=0.00001 --epoch=200 --batch_size=$BATCH_SIZE
