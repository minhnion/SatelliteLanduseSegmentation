# BATCH_SIZE=32
# python main.py --cuda --model=ViTUnet --gpu_id=0 --lr=0.01 --epoch=200 --depth=12 --heads=4 --dropout=0.2 --batch_size=$BATCH_SIZE
# python main.py --cuda --model=ViTUnet --gpu_id=0 --lr=0.005 --epoch=200 --depth=12 --heads=8 --dropout=0.2 --batch_size=$BATCH_SIZE
# python main.py --cuda --model=ViTUnet --gpu_id=0 --lr=0.01 --epoch=200 --depth=12 --heads=4 --dropout=0 --batch_size=$BATCH_SIZE
# python main.py --cuda --model=ViTUnet --gpu_id=0 --lr=0.01 --epoch=200 --depth=12 --heads=4 --dropout=0.1 --batch_size=$BATCH_SIZE
# python main.py --cuda --model=ViTUnet --gpu_id=0 --lr=0.01 --epoch=200 --depth=13 --heads=4 --dropout=0.1 --batch_size=$BATCH_SIZE
# python main.py --cuda --model=ViTUnet --gpu_id=0 --lr=0.01 --epoch=200 --depth=14 --heads=4 --dropout=0.1 --batch_size=$BATCH_SIZE
# python main.py --cuda --model=ViTUnet --gpu_id=0 --lr=0.01 --epoch=200 --depth=15 --heads=4 --dropout=0.1 --batch_size=$BATCH_SIZE
# python main.py --cuda --model=ViTUnet --gpu_id=0 --lr=0.01 --epoch=200 --depth=16 --heads=4 --dropout=0.1 --batch_size=$BATCH_SIZE
# python main.py --cuda --model=ViTUnet --gpu_id=0 --lr=0.01 --epoch=200 --depth=8 --heads=3 --dropout=0.3 --batch_size=$BATCH_SIZE
# python main.py --cuda --model=ViTUnet --gpu_id=0 --lr=0.01 --epoch=200 --depth=8 --heads=2--dropout=0.3 --batch_size=$BATCH_SIZE
# python main.py --cuda --model=ViTUnet --gpu_id=0 --lr=0.01 --epoch=200 --depth=8 --heads=4 --dropout=0.3 --batch_size=$BATCH_SIZE
# python main.py --cuda --model=ViTUnet --gpu_id=0 --lr=0.01 --epoch=200 --depth=8 --heads=6 --dropout=0.3 --batch_size=$BATCH_SIZE

BATCH_SIZE=2
python main.py --cuda --model=ViTUnet --gpu_id=0 --lr=0.00001 --epoch=10 --batch_size=$BATCH_SIZE
