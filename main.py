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
from model.CrossAttentionUNetSR.model import CrossAttentionUNetSR
from model.FCNResNet.model import FCNResnet
from model.ESRT.model import ESRT

import torch.optim as optim
import torch.nn as nn
from datetime import datetime
from utils.loss import DiceLoss
from layers.unet_layers import *

from utils.trainer import train, train_sr_seg
from utils.evaluator import evaluate_on_test_set, evaluate_sr_seg_on_test_set
import os
import wandb
from dotenv import load_dotenv
import warnings
from rasterio.errors import NotGeoreferencedWarning

import shutil

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":
    try:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
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

        root_log_path = '/mnt/anhtn/log'
        model_log_path = f'dataset:{base_path}/model:{args.model}/datetime:{timestamp}_epoch:{args.epoch}_bs:{args.batch_size}_lr:{args.lr}/'
        weight_path = os.path.join(root_log_path, 'weights', model_log_path)
        image_path = os.path.join(root_log_path, 'images', model_log_path)

        if not os.path.exists(weight_path):
            os.makedirs(weight_path)
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        rgb_only = False
        model = None
        sr_output = True
        sr_cat = False
        mask_scale=None
        scale_factor = 2

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
            model = UNetSR(n_classes=n_classes, n_channels=n_channels, sr_cat=sr_cat, sr_output=sr_output).to(device)
            mask_scale = 2 if sr_output else None
        elif args.model == 'CrossAttentionUNetSR':
            model = CrossAttentionUNetSR(n_classes=n_classes, n_channels=n_channels).to(device)
        elif args.model == 'FCNResNet':
            model = FCNResnet(n_channels=n_channels, n_classes=n_classes).to(device)
        elif args.model == 'ESSRT':
            model = ESRT(n_channels=n_channels, n_classes=n_classes, upscale=scale_factor).to(device)

        if args.pretrained:
            checkpoint = args.pretrained
            model.load_state_dict(torch.load(checkpoint),strict=False)

        model = model.to(device)

        print("model loaded")
        if args.model == 'ESSRT':
            train_loader, val_loader, test_loader = load_dataloader(batch_size=args.batch_size, root_dir=root_dir, size=size, rgb_only=rgb_only, mask_scale=mask_scale, scale_factor=scale_factor)
        else:
            train_loader, val_loader, test_loader = load_dataloader(batch_size=args.batch_size, root_dir=root_dir, size=size, rgb_only=rgb_only, mask_scale=mask_scale)
        print("dataloader loaded")

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        weight = torch.Tensor([0.3, 1.0, 1.5, 1.5, 1.0]).to(device)
        criterion_seg = nn.CrossEntropyLoss(weight=weight)
        criterion_sr = nn.L1Loss()
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

        num_epochs = 1 if base_path == 'dummy_dataset' else args.epoch

        if args.model != 'ESSRT':
            model = train(model, train_loader, val_loader, optimizer, scheduler, criterion_seg, classes, device, l1_lambda=0, num_epochs=num_epochs, save_path=weight_path + 'weight.pth', image_dir=image_path, early_stop=True, patience=30)
            evaluate_on_test_set(model, test_loader, classes, image_dir=image_path, num_samples=1)
        else:
            model = train_sr_seg(model, train_loader, val_loader, optimizer, scheduler, criterion_seg, criterion_sr, classes, device, l1_lambda=0, num_epochs=num_epochs, save_path=weight_path + 'weight.pth', image_dir=image_path, early_stop=True, patience=30)
            evaluate_sr_seg_on_test_set(model, test_loader, classes, image_dir=image_path, num_samples=1)

        run.finish()

    except Exception as e:
        print(f"An error occurred: {e}")
