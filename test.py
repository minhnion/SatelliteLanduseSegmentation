from tqdm import tqdm
import torch
from utils.seed import set_seed
from utils.parser import parse_args
from data.dataloader import load_dataloader, load_one_dataloader

from model.ViT.model import UNet
from model.LinkNet.model import LinkNet
from model.PromptedVitUnet.model import PromptedVitUnet
from model.PretrainedViT.model import PretrainedViT
from model.PretrainedViTUNet.model import PretrainedViTUNet
from model.UNetSR.model import UNetSR

import torch.optim as optim
import torch.nn as nn
from datetime import datetime
from utils.loss import DiceLoss
from layers.unet_layers import *

from utils.trainer import train
from utils.evaluator import evaluate_on_test_set
import os
import wandb
from dotenv import load_dotenv
import warnings
from rasterio.errors import NotGeoreferencedWarning

import shutil

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

if __name__ == "__main__":
    try:
        # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        # Load environment variables from .env file
        load_dotenv()

        # Initialize
        set_seed(20)
        args = parse_args()
        device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
        print(f"Using device: {device}")

        WANDB_API_KEY = os.getenv('WANDB_API_KEY')
        wandb.login(key=WANDB_API_KEY)

        CFG = dict(
            optimiser='Adam',
            learning_rate=args.lr,
            batch_size=args.batch_size,
            epochs=args.epoch,
        )
        classes = ['unidentifiable', 'forest', 'rice_field', 'water', 'residential']
        n_classes = len(classes)
        # base_path = 'sentinel_dataset'
        base_path = 'new_13bands_dataset_splitted'
        n_channels = 13 # for new dataset, n_channels = 4
        root_dir = f'/mnt/henryng/{base_path}'

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        weight_path = f"/mnt/anhtn/log/weights/dataset:{base_path}/model:{args.model}/epoch:{args.epoch}_bs:{args.batch_size}_lr:{args.lr}_datetime:{timestamp}/"
        image_path =  f"/mnt/anhtn/log/images/dataset:{base_path}/model:{args.model}/epoch:{args.epoch}_bs:{args.batch_size}_lr:{args.lr}_datetime:{timestamp}/"
        if not os.path.exists(weight_path):
            os.makedirs(weight_path)
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        rgb_only = False
        model = None
        sr_cat = False

        h_w = 128
        size = (h_w, h_w)
        if args.model == 'LinkNet':
            model = LinkNet(n_classes=n_classes, n_channels=n_channels).to(device)
        elif args.model == 'ViTUnet':
            model = UNet(n_classes=n_classes, n_channels=n_channels, depth=args.depth, heads=args.heads, dropout=args.dropout).to(device)
        elif args.model == 'PretrainedViT':
            size = (224, 224)
            model = PretrainedViT(n_classes=n_classes).to(device)
        elif args.model == 'PretrainedViTUNet':
            model = PretrainedViTUNet(n_channels=5, n_classes=n_classes)
            model.load_state_dict(torch.load('model/PretrainedViTUNet/best_pretrained_13bands_100e_model.pth'))
            model.inc = DoubleConv(n_channels, 64)
            model.to(device)
        elif args.model == 'PromptedViTUnet':
            model = PromptedVitUnet(n_classes=n_classes, n_channels=n_channels, depth=args.depth, heads=args.heads, dropout=args.dropout).to(device)
        elif args.model == 'UNetSR':
            model = UNetSR(n_classes=n_classes, n_channels=n_channels, sr_cat=sr_cat).to(device)
            checkpoint = 'model/ViT/current_pretrained_unet_model.pth'
            model.load_state_dict(torch.load(checkpoint),strict=False)

        print("model loaded")
        train_loader, val_loader, test_loader = load_dataloader(batch_size=args.batch_size, root_dir=root_dir, size=size, rgb_only=rgb_only)
        print("dataloader loaded")

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        # weight = torch.Tensor([0.0, 1.0, 1.0, 1.5, 1.0]).to(device)
        # criterion = DiceLoss(weight=weight)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        torch.cuda.empty_cache()

        run = wandb.init(
            entity='mingixpt-hust',
            project=os.path.basename(root_dir),
            config=CFG,
            save_code=True,
            job_type='train',
            group = f'{args.model}',
            name=f'model:{args.model}_epoch:{args.epoch}_bs:{args.batch_size}_lr:{args.lr}_cat?:{sr_cat}_datetime:{timestamp}'
        )
        # model = train(model, train_loader, val_loader, optimizer, scheduler, criterion, classes, device, num_epochs=args.epoch, save_path=weight_path + 'best_weight.pth', image_dir=image_path, early_stop=True, patience=20)
        evaluate_on_test_set(model, test_loader, classes, image_dir=image_path)

        # model_path = '/mnt/anhtn/log/weights/dataset:gg_earth_cut64_dataset_model:ViTUnet_epoch:200_bs:16_lr:0.0001_datetime:20241130_015542/best_weight.pth'
        # model.load_state_dict(torch.load(model_path, weights_only=True))
        # infer_loader = load_one_dataloader(batch_size=args.batch_size, root_dir=root_dir)
        # evaluate_on_test_set(model, infer_loader, classes, image_dir=image_path)
        run.finish()

    except Exception as e:
        print(f"An error occurred: {e}")
        if os.path.exists(weight_path):
            shutil.rmtree(weight_path)
        if os.path.exists(image_path):
            shutil.rmtree(image_path)
        raise
