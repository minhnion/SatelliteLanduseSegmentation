import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm

from model.FoundationKDNet.models import FoundationKDModel, StudentModel
from model.FoundationKDNet.loss import WassersteinLoss, DiceLossSoftmax, DiceLossArgmax, PairWiseLoss
from data.dataloader import load_dataloader
from dataset_config.load_config import load_config

# Utility function to print the number of parameters in a model
def print_model_parm_nums(model, name):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{name} has {total_params / 1e6:.2f}M parameters')

class KnowledgeDistillationModel:
    # Return the name of the model
    def name(self):
        return 'kd_seg_sr'

    def __init__(self, args):
        self.args = args
        self.device = args.device

        # Initialize teacher and student models
        self.teacher = FoundationKDModel(args.n_channels, args.n_classes, embed_dim=args.embed_dim).to(self.device)
        self.student = StudentModel(args.n_channels, args.n_classes, embed_dim=args.embed_dim).to(self.device)

        # Load pretrained teacher if checkpoint is provided
        if args.T_ckpt_path and os.path.exists(args.T_ckpt_path):
            self.teacher.load_state_dict(torch.load(args.T_ckpt_path))
            print(f"Loaded teacher checkpoint from {args.T_ckpt_path}")

        # Freeze teacher model parameters
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        # Initialize optimizer for student model
        self.G_solver = optim.SGD(
            self.student.parameters(),
            lr=args.lr_g,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )

        # Initialize loss functions
        self.criterion_ce = nn.CrossEntropyLoss().to(self.device)
        self.criterion_wasserstein = WassersteinLoss().to(self.device)
        self.criterion_dice_am = DiceLossArgmax().to(self.device)
        self.criterion_l1 = nn.L1Loss().to(self.device)
        self.criterion_pairwise = PairWiseLoss().to(self.device)  # Pair-wise loss for feature maps

        # Dictionary to track loss history for visualization
        self.loss_history = {
            'total_loss': [], 'sr_loss_ts': [], 'seg_loss_ts': [], 'sr_loss_gt': [],
            'seg_loss_gt': [], 'embedding_loss': [], 'pairwise_loss': []
        }

        # Create directory for saving snapshots
        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

    def set_input(self, data):
        """Set input data for training or evaluation"""
        self.images, self.labels, self.lr_images, _, _, _ = data
        self.images = self.images.to(self.device)
        self.labels = self.labels.long().to(self.device)
        self.lr_images = self.lr_images.to(self.device)

    def forward(self):
        """Perform forward pass for teacher and student models"""
        with torch.no_grad():
            feature_maps_t, embedding_t, (sr_map_t, seg_map_t) = self.teacher(self.images)
        feature_maps_s, embedding_s, (sr_map_s, seg_map_s) = self.student(self.images)

        self.teacher_outputs = (sr_map_t, seg_map_t)
        self.teacher_feature_maps = feature_maps_t
        self.teacher_embedding = embedding_t
        self.student_outputs = (sr_map_s, seg_map_s)
        self.student_feature_maps = feature_maps_s
        self.student_embedding = embedding_s

    def backward(self):
        """Compute losses and perform backpropagation for student"""
        self.G_solver.zero_grad()

        # Task-specific losses (student vs teacher)
        sr_loss_ts = self.criterion_wasserstein(self.student_outputs[0], self.teacher_outputs[0])
        seg_wast_loss_ts = self.criterion_wasserstein(self.student_outputs[1], self.teacher_outputs[1])
        seg_ce_loss_ts = self.criterion_ce(self.student_outputs[1], self.teacher_outputs[1].argmax(dim=1))
        seg_loss_ts = seg_ce_loss_ts + seg_wast_loss_ts
        task_loss_ts = sr_loss_ts + seg_loss_ts

        # Task-specific losses (student vs ground truth)
        sr_loss_gt = self.criterion_l1(self.student_outputs[0], self.lr_images)
        seg_ce_loss_gt = self.criterion_ce(self.student_outputs[1], self.labels)
        seg_dice_loss_gt = self.criterion_dice_am(self.student_outputs[1], self.labels)
        seg_loss_gt = self.args.beta * seg_ce_loss_gt + (1 - self.args.beta) * seg_dice_loss_gt  # Weighted CE and Dice
        task_loss_gt = sr_loss_gt + seg_loss_gt

        # Combined task loss
        task_loss = task_loss_ts + task_loss_gt

        # Embedding loss
        embedding_loss = self.criterion_wasserstein(self.student_embedding, self.teacher_embedding)

        # Pair-wise loss on feature maps (averaged over all feature maps)
        pairwise_loss = 0
        for s_fm, t_fm in zip(self.student_feature_maps, self.teacher_feature_maps):
            pairwise_loss += self.criterion_pairwise(s_fm, t_fm)
        pairwise_loss = pairwise_loss / len(self.student_feature_maps)

        # Total loss with weighted components
        G_loss = (1 - self.args.alpha) * task_loss + self.args.alpha * embedding_loss + self.args.lambda_pairwise * pairwise_loss

        G_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        self.G_solver.step()

        # Update loss tracking variables
        self.sr_G_loss = sr_loss_ts.item() + sr_loss_gt.item()
        self.seg_G_loss = seg_loss_ts.item() + seg_loss_gt.item()
        self.G_loss = G_loss.item()

        # Append losses to history for visualization
        self.loss_history['total_loss'].append(G_loss.item())
        self.loss_history['sr_loss_ts'].append(sr_loss_ts.item())
        self.loss_history['seg_loss_ts'].append(seg_loss_ts.item())
        self.loss_history['sr_loss_gt'].append(sr_loss_gt.item())
        self.loss_history['seg_loss_gt'].append(seg_loss_gt.item())
        self.loss_history['embedding_loss'].append(embedding_loss.item())
        self.loss_history['pairwise_loss'].append(pairwise_loss.item())

        return {
            'total_loss': G_loss.item(),
            'sr_loss_ts': sr_loss_ts.item(),
            'seg_loss_ts': seg_loss_ts.item(),
            'sr_loss_gt': sr_loss_gt.item(),
            'seg_loss_gt': seg_loss_gt.item(),
            'embedding_loss': embedding_loss.item(),
            'pairwise_loss': pairwise_loss.item()
        }

    def optimize_parameters(self):
        """Perform one training step"""
        self.forward()
        return self.backward()

    def print_info(self, epoch, step):
        """Print training information to log"""
        logging.info(f'Epoch: {epoch}, Step: {step}, G_lr: {self.G_solver.param_groups[-1]["lr"]:.6f}, '
                     f'G_loss: {self.G_loss:.5f} (SR: {self.sr_G_loss:.5f}, Seg: {self.seg_G_loss:.5f}, '
                     f'Embedding: {self.loss_history["embedding_loss"][-1]:.5f}, Pairwise: {self.loss_history["pairwise_loss"][-1]:.5f})')

    def save_ckpt(self, epoch, step, metrics):
        """Save model checkpoint"""
        torch.save(self.student.state_dict(), os.path.join(
            self.args.snapshot_dir, f'kd_seg_sr_epoch_{epoch}_step_{step}_loss_{metrics["total_loss"]:.5f}.pth'
        ))

    def evaluate(self, val_loader):
        """Evaluate the model on validation set"""
        self.student.eval()
        total_sr_loss = 0
        total_seg_loss = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                self.set_input(data)
                self.forward()
                sr_loss = self.criterion_wasserstein(self.student_outputs[0], self.teacher_outputs[0])
                seg_loss = self.criterion_ce(self.student_outputs[1], self.teacher_outputs[1].argmax(dim=1))
                total_sr_loss += sr_loss.item()
                total_seg_loss += seg_loss.item()
                total += 1
        self.student.train()
        return {
            'sr_loss': total_sr_loss / total,
            'seg_loss': total_seg_loss / total,
            'total_loss': (total_sr_loss + total_seg_loss) / total
        }

    def visualize_losses(self, save_dir='loss_plots'):
        """Visualize all tracked losses across entire training and save plot"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.figure(figsize=(12, 8))
        for loss_name, loss_values in self.loss_history.items():
            plt.plot(loss_values, label=loss_name)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Loss Curves - Full Training')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'loss_full_training.png'))
        plt.close()

# Training function with progress bar
def train_kd(model, train_loader, val_loader, num_epochs=10):
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.student.train()
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for step, data in enumerate(train_bar):
            model.set_input(data)
            metrics = model.optimize_parameters()
            if step % 10 == 0:
                model.print_info(epoch, step)
                train_bar.set_postfix({
                    'G_loss': f'{metrics["total_loss"]:.5f}',
                    'SR': f'{metrics["sr_loss_ts"] + metrics["sr_loss_gt"]:.5f}',
                    'Seg': f'{metrics["seg_loss_ts"] + metrics["seg_loss_gt"]:.5f}',
                    'Embed': f'{metrics["embedding_loss"]:.5f}',
                    'Pair': f'{metrics["pairwise_loss"]:.5f}'
                })
        # Evaluate and save checkpoint (no visualization here)
        metrics = model.evaluate(val_loader)
        model.save_ckpt(epoch, step, metrics)
        logging.info(f'Epoch {epoch} Validation Metrics: {metrics}')

    # Visualize all losses at the end of training
    model.visualize_losses()

# Argument parsing and main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Distillation for Segmentation and SR")
    parser.add_argument('--n_channels', type=int, default=13, help='Number of input channels')
    parser.add_argument('--n_classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--embed_dim', type=int, default=1024, help='Embedding dimension')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--lr_g', type=float, default=1e-6, help='Learning rate for generator')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--snapshot_dir', type=str, default='./snapshots', help='Directory to save checkpoints')
    parser.add_argument('--T_ckpt_path', type=str, default=None, help='Teacher checkpoint path')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for embedding loss')
    parser.add_argument('--beta', type=float, default=0.5, help='Weight for CE vs Dice loss')
    parser.add_argument('--lambda_pairwise', type=float, default=0.1, help='Weight for pairwise loss')
    args = parser.parse_args()

    # Load dataset configuration and dataloaders
    config = load_config("dataset_config/config.yaml")  # Assumes config file exists
    train_loader = load_dataloader(config, 'train', batch_size=args.batch_size)
    val_loader = load_dataloader(config, 'val', batch_size=args.batch_size)

    # Initialize model and start training
    model = KnowledgeDistillationModel(args)
    train_kd(model, train_loader, val_loader, num_epochs=10)
