from tqdm import tqdm
import torch
from utils.seed import set_seed
from utils.parser import parse_args
from data.dataloader import load_dataloader, load_one_dataloader
from model.ViT.model import UNet
from model.LinkNet.model import LinkNet
from model.PromptedVitUnet.model import PromptedVitUnet
import torch.optim as optim
import torch.nn as nn
from datetime import datetime

from utils.trainer import train
from utils.evaluator import evaluate_on_test_set
import os
import wandb
from dotenv import load_dotenv

if __name__ == "__main__":
    try:
        load_dotenv()

        # Initialize
        set_seed(20)
        args = parse_args()
        device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
        print(f"Using device: {device}")

        base_path = 'sentinel_cut1024_dataset/train'
        root_dir = f'/mnt/henryng/{base_path}'

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path =  f"/mnt/anhtn/log/images/dataset:{base_path}/model:{args.model}_epoch:{args.epoch}_bs:{args.batch_size}_lr:{args.lr}_datetime:{timestamp}/"

        if not os.path.exists(image_path):
            os.makedirs(image_path)

        classes = ['unidentifiable', 'forest', 'rice_field', 'water', 'residential']
        n_channels = 4 # for new dataset, n_channels = 4
        n_classes = len(classes)

        if args.model == 'LinkNet':
            model = LinkNet(n_classes=n_classes, n_channels=n_channels).to(device)
        elif args.model == 'ViTUnet':
            model = UNet(n_classes=n_classes, n_channels=n_channels, depth=args.depth, heads=args.heads, dropout=args.dropout).to(device)
        elif args.model == 'PromptedViTUnet':
            model = PromptedVitUnet(n_classes=n_classes, n_channels=n_channels, depth=args.depth, heads=args.heads, dropout=args.dropout).to(device)

        torch.cuda.empty_cache()

        model_path = '/mnt/anhtn/log/weights/dataset:sentinel_dataset_cut256_filtered_model:ViTUnet_epoch:200_bs:32_lr:0.0001_datetime:20241201_224952/best_weight.pth'
        model.load_state_dict(torch.load(model_path, weights_only=True))
        infer_loader = load_one_dataloader(batch_size=args.batch_size, root_dir=root_dir)
        evaluate_on_test_set(model, infer_loader, classes, image_dir=image_path, wandb_setup=False)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise
