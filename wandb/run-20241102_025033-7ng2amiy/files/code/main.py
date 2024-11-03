from tqdm import tqdm
import torch
from utils.seed import set_seed
from utils.parser import parse_args
from data.dataloader import load_dataloader
from model.ViT.model import UNet
from model.LinkNet.model import LinkNet
import torch.optim as optim
import torch.nn as nn

from utils.trainer import train
from utils.evaluator import evaluate_on_test_set
import os
import wandb
from dotenv import load_dotenv

if __name__ == "__main__":
    try:
        # Load environment variables from .env file
        load_dotenv()

        WANDB_API_KEY = os.getenv('WANDB_API_KEY')
        wandb.login(key=WANDB_API_KEY)

        classes = ['unidentifiable', 'forest', 'rice_field', 'water', 'residential']
        n_channels = 5
        n_classes = len(classes)

        set_seed(20)
        args = parse_args()
        device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
        print(f"Using device: {device}")
        train_loader, val_loader, test_loader = load_dataloader(batch_size=args.batch_size)
        model = None

        if args.model == 'linknet':
            model = LinkNet(n_classes=n_classes, n_channels=n_channels).to(device)
        elif args.model == 'ViTUnet':
            model = UNet(n_classes=n_classes, n_channels=n_channels).to(device)
        elif args.model == 'PromptedViTUnet':
            model = UNet(n_classes=n_classes, n_channels=n_channels, prompted=True).to(device)

        CFG = dict(
            optimiser='Adam',
            learning_rate=args.lr,
            batch_size=args.batch_size,
            epochs=args.epoch,
        )

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        run = wandb.init(
            entity='mingixpt-hust',
            project='landuse-full-cut-data',
            config=CFG,
            save_code=True,
            job_type='train'
        )

        train(model, train_loader, val_loader, optimizer, scheduler, criterion, classes, device, num_epochs=args.epoch, save_path='best_model.pth', early_stop=True, patience=20)

        run.finish()

        evaluate_on_test_set(model, test_loader, classes)
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
