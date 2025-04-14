import torch
from utils.seed import set_seed
from utils.parser import parse_args
from data.dataloader import load_dataloader, load_hr_dataloader

from model.ViT.model import UNet
from model.LinkNet.model import LinkNet
from model.PromptedVitUnet.model import PromptedVitUnet
from model.PretrainedViT.model import PretrainedViT
from model.PretrainedViTUNet.model import PretrainedViTUNet
from model.UNetSR.model import UNetSR
from model.CrossAttentionUNetSR.model import CrossAttentionUNetSR
from model.FCNResNet.model import FCNResnet
from model.ESRT.model import ESRT
from model.Foundation.model import FoundationModel
from model.FoundationKDNet.models import FoundationKDModel
from model.SCNet.model import SCNet
from model.LSKNet.model import LSKNetSegmentor
from model.PyramidMamba.model import PyramidMamba

import torch.optim as optim
import torch.nn as nn
from datetime import datetime
from layers.unet_layers import *

from utils.trainer import train, train_sr_seg, train_scnet
from utils.evaluator import evaluate_on_test_set, evaluate_sr_seg_on_test_set, evaluate_scnet_on_test_set
from utils.model_utils import init_weights_he
import os
import wandb
from dotenv import load_dotenv
import warnings
from rasterio.errors import NotGeoreferencedWarning
import traceback
from dataset_config.load_config import load_config

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":
    sr_seg_models = ['ESSRT', 'FoundationModel', 'FoundationKDModel', 'FoundationKDModelHR', 'UNetSR', 'CrossAttentionUNetSR', 'SCNet', 'LSKNet', 'PyramidMamba']
    try:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        # Load environment variables from .env file
        load_dotenv()

        # Initialize
        set_seed(20)
        args = parse_args()
        # device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
        device = "cpu"
        print(f"Using device: {device}")

        WANDB_API_KEY = os.getenv('WANDB_API_KEY')
        wandb.login(key=WANDB_API_KEY)

        CFG = dict(
            optimiser='Adam',
            learning_rate=args.lr,
            batch_size=args.batch_size,
            epochs=args.epoch,
        )

        # Load dataset config
        dataset = args.dataset
        dataset_module = f'dataset_config.{dataset}'
        dataset_config = load_config(dataset_module)
        RGB_TO_CLASSES = dataset_config['classes']
        CLASSES_TO_RGB = {}

        for k, v in RGB_TO_CLASSES.items():
            if v not in CLASSES_TO_RGB:
                CLASSES_TO_RGB[v] = k

        classes = []
        for cls in RGB_TO_CLASSES.values():
            if cls not in classes:
                classes.append(cls)
        n_classes = dataset_config['n_classes']
        n_channels = dataset_config['n_channels']
        base_path = dataset_config['base_path']
        num_tiles = dataset_config['num_tiles']
        weights = dataset_config['weights']

        # base_path = 'new_13bands_dataset_splitted'
        root_dir = f'/mnt/hungvv/minh'
        dataset_dir = os.path.join(root_dir,'dataset', base_path) if not args.data_path else args.data_path
        log_path = os.path.join(root_dir, 'log') if not args.log_path else args.log_path

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_log_path = f'dataset:{base_path}/model:{args.model}/datetime:{timestamp}_epoch:{args.epoch}_bs:{args.batch_size}_lr:{args.lr}/'
        weight_path = os.path.join(log_path, 'weights', model_log_path)
        image_path = os.path.join(log_path, 'images', model_log_path)

        if not os.path.exists(weight_path):
            os.makedirs(weight_path)
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        model = None
        sr_output = True
        sr_cat = False
        scale_factor = 2
        mask_scale = None

        h_w = 1024
        size = (h_w, h_w)

        model_dict = {
            'LinkNet': lambda: LinkNet(n_classes=n_classes, n_channels=n_channels).to(device),
            'ViTUnet': lambda: UNet(n_classes=n_classes, n_channels=n_channels, depth=args.depth, heads=args.heads, dropout=args.dropout).to(device),
            'PretrainedViT': lambda: PretrainedViT(n_classes=n_classes).to(device),
            'PretrainedViTUNet': lambda: PretrainedViTUNet(n_channels=5, n_classes=n_classes).to(device),
            'PromptedViTUnet': lambda: PromptedVitUnet(n_classes=n_classes, n_channels=n_channels, depth=args.depth, heads=args.heads, dropout=args.dropout).to(device),
            'UNetSR': lambda: UNetSR(n_classes=n_classes, n_channels=n_channels, sr_cat=sr_cat, sr_output=sr_output).to(device),
            'CrossAttentionUNetSR': lambda: CrossAttentionUNetSR(n_classes=n_classes, n_channels=n_channels).to(device),
            'FCNResNet': lambda: FCNResnet(n_channels=n_channels, n_classes=n_classes).to(device),
            'ESSRT': lambda: ESRT(n_channels=n_channels, n_classes=n_classes, upscale=scale_factor).to(device),
            'FoundationModel': lambda: FoundationModel(n_channels=n_channels, n_classes=n_classes, upscale_factor=scale_factor).to(device),
            'SCNet': lambda: SCNet(in_channels_Y=n_channels, in_channels_alpha=5 if n_channels==13 else 4, n_classes=n_classes, scale_factor=scale_factor).to(device),
            'LSKNet': lambda: LSKNetSegmentor(n_classes=n_classes, n_channels=n_channels).to(device),
            'PyramidMamba': lambda: PyramidMamba(n_classes=n_classes, n_channels=n_channels, img_size=h_w).to(device),
            'FoundationKDModel': lambda: FoundationKDModel(n_channels=n_channels, n_classes=n_classes, upscale_factor=scale_factor).to(device),
            'FoundationKDModelHR': lambda: FoundationKDModel(n_channels=n_channels, n_classes=n_classes, upscale_factor=16, sr_out_chan=3).to(device),
        }

        model = model_dict.get(args.model, lambda: None)()
        if args.model == 'UNetSR':
            sr_output = True
            mask_scale = scale_factor if sr_output else None

        if args.pretrained:
            checkpoint = args.pretrained
            pretrained_dict = torch.load(checkpoint)
            model_dict = model.state_dict()
            # Filter out unnecessary keys and initialize mismatched layers
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
            mismatched_layers = {k: v for k, v in model_dict.items() if k not in pretrained_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            # Initialize mismatched layers with He initialization
            # for layer_name in mismatched_layers:
            #     if 'weight' in layer_name:
            #         nn.init.kaiming_normal_(model.state_dict()[layer_name], mode='fan_out', nonlinearity='relu')
            #     elif 'bias' in layer_name:
            #         nn.init.constant_(model.state_dict()[layer_name], 0)
        # else:
        #     model.apply(init_weights_he)

        model = model.to(device)

        print("model loaded")
        if args.model =='FoundationKDModelHR':
            train_loader, val_loader, test_loader = load_hr_dataloader(batch_size=args.batch_size, classes=classes, RGB_TO_CLASSES=RGB_TO_CLASSES, root_dir=dataset_dir, size=size, num_tiles=num_tiles, scale_factor=16)
        elif args.model in sr_seg_models:
            print("Sr + Seg model")
            train_loader, val_loader, test_loader = load_dataloader(batch_size=args.batch_size, classes=classes, RGB_TO_CLASSES=RGB_TO_CLASSES, root_dir=dataset_dir, size=size, mask_scale=mask_scale, num_tiles=num_tiles, scale_factor=scale_factor)
        else:
            train_loader, val_loader, test_loader = load_dataloader(batch_size=args.batch_size, classes=classes, RGB_TO_CLASSES=RGB_TO_CLASSES, root_dir=dataset_dir, size=size, mask_scale=mask_scale, num_tiles=num_tiles)
        print("dataloader loaded")

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # weight = torch.Tensor([0.3, 1.0, 1.5, 1.5, 1.0]).to(device)
        # criterion_seg = nn.CrossEntropyLoss(weight=weight)
        criterion_seg = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))
        criterion_sr = nn.L1Loss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        torch.cuda.empty_cache()

        run = wandb.init(
            entity='mingixpt-hust',
            project=os.path.basename(dataset_dir),
            config=CFG,
            save_code=True,
            job_type='train',
            group = f'{args.model}',
            name=f'model:{args.model}_epoch:{args.epoch}_bs:{args.batch_size}_lr:{args.lr}_cat?:{sr_cat}_datetime:{timestamp}'
        )

        num_epochs = 1 if base_path == 'dummy_dataset' else args.epoch

        if args.model in sr_seg_models:
            # model = train_sr_seg(model, train_loader, val_loader, optimizer, scheduler, criterion_seg, criterion_sr, classes, device, l1_lambda=0, num_epochs=num_epochs, save_path=weight_path + 'weight.pth', image_dir=image_path, early_stop=False, patience=30)
            evaluate_sr_seg_on_test_set(model, test_loader, classes, CLASSES_TO_RGB, image_dir=image_path, num_samples=1)
        elif args.model == 'SCNet':
            # model = train_scnet(model, train_loader, val_loader, optimizer, scheduler, criterion_seg, classes, device, l1_lambda=0, num_epochs=num_epochs, save_path=weight_path + 'weight.pth', image_dir=image_path, early_stop=False, patience=30)
            evaluate_scnet_on_test_set(model, test_loader, classes, CLASSES_TO_RGB, image_dir=image_path, num_samples=1)
        else:
            # model = train(model, train_loader, val_loader, optimizer, scheduler, criterion_seg, classes, device, l1_lambda=0, num_epochs=num_epochs, save_path=weight_path + 'weight.pth', image_dir=image_path, early_stop=False, patience=30)
            evaluate_on_test_set(model, test_loader, classes, CLASSES_TO_RGB, image_dir=image_path, num_samples=1)

        run.finish()

    except Exception as e:
        print("An error occurred:", str(e))
        traceback.print_exc()
