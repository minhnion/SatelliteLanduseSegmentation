from tqdm import tqdm
import torch
from utils.seed import set_seed
from utils.parser import parse_args
from data.dataloader import load_dataloader
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

        run = wandb.init(
            entity='mingixpt-hust',
            project='landuse-fixed',
            config=CFG,
            save_code=True,
            job_type='train',
            group = f'{args.model}',
            name=f'model:{args.model}_epoch:{args.epoch}_bs:{args.batch_size}_lr:{args.lr}'
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        weight_path = f"./log/weights/model:{args.model}_epoch:{args.epoch}_bs:{args.batch_size}_lr:{args.lr}_datetime:{timestamp}/"
        image_path =  f"./log/images/model:{args.model}_epoch:{args.epoch}_bs:{args.batch_size}_lr:{args.lr}_datetime:{timestamp}/"
        if not os.path.exists(weight_path):
            os.makedirs(weight_path)
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        classes = ['unidentifiable', 'forest', 'rice_field', 'water', 'residential']
        root_dir = '/mnt/henryng/sentinel_dataset_cut256_filtered'
        n_channels = 5
        n_classes = len(classes)

        train_loader, val_loader, test_loader = load_dataloader(batch_size=args.batch_size, root_dir=root_dir)
        model = None

        if args.model == 'linknet':
            model = LinkNet(n_classes=n_classes, n_channels=n_channels).to(device)
        elif args.model == 'ViTUnet':
            model = UNet(n_classes=n_classes, n_channels=n_channels).to(device)
        elif args.model == 'PromptedViTUnet':
            model = PromptedVitUnet(n_classes=n_classes, n_channels=n_channels, depth=6, heads=4, dropout=0.3).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        torch.cuda.empty_cache()
        
        train(model, train_loader, val_loader, optimizer, scheduler, criterion, classes, device, num_epochs=args.epoch, save_path=weight_path + 'best_weight.pth', image_dir=image_path, early_stop=True, patience=20)

        evaluate_on_test_set(model, test_loader, classes, image_path=image_path)
        run.finish()

    except Exception as e:
        print(f"An error occurred: {e}")
        raise
