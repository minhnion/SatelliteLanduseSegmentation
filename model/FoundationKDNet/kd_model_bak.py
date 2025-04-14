import torch
import torch.nn as nn
from torch.optim import Adam

from model.FoundationKDNet.models import FoundationKDModel, StudentModel
from model.FoundationKDNet.loss import WassersteinLoss, DiceLossSoftmax, DiceLossArgmax

class KnowledgeDistillationModel:
    def __init__(self, n_channels=13, n_classes=5, embed_dim=1024, device='cuda'):
        self.device = device

        self.teacher = FoundationKDModel(n_channels, n_classes, embed_dim=embed_dim).to(device)
        self.student = StudentModel(n_channels, n_classes, embed_dim=embed_dim).to(device)

        # Freeze teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        # Initialize optimizers
        self.optimizer = Adam(self.student.parameters(), lr=1e-6)

        # Initialize loss functions
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_wasserstein = WassersteinLoss()
        self.criterion_dice_sm = DiceLossSoftmax()
        self.criterion_dice_am = DiceLossArgmax()
        self.criterion_cosine = nn.CosineEmbeddingLoss()

    def set_input(self, data):
        """Set input data"""
        if isinstance(data, (tuple, list)):
            self.images = data[0].to(self.device)
            if len(data) > 1:
                self.labels = data[1].to(self.device)
                self.lr_images = data[2].to(self.device)
        else:
            self.images = data.to(self.device)

    def forward(self):
        """Forward pass"""
        with torch.no_grad():
            # Teacher forward pass
            feature_maps_t , embedding_t, (sr_map_t, seg_map_t) = self.teacher(self.images)

        # Student forward pass
        feature_maps_s, embedding_s, (sr_map_s, seg_map_s) = self.student(self.images)

        # Store outputs for loss computation
        self.teacher_outputs = (sr_map_t, seg_map_t)
        self.teacher_feature_maps = feature_maps_t
        self.teacher_embedding = embedding_t

        self.student_outputs = (sr_map_s, seg_map_s)
        self.student_feature_maps = feature_maps_s
        self.student_embedding = embedding_s

        # print(f"\nTeacher sr_map shape: {sr_map_t.shape}")
        # print(f"Teacher seg_map shape: {seg_map_t.shape}")
        # print(f"Teacher embedding shape: {embedding_t.shape}\n")
        # print(f"Student sr_map shape: {sr_map_s.shape}")
        # print(f"Student seg_map shape: {seg_map_s.shape}")
        # print(f"Student embedding shape: {embedding_s.shape}\n")

        # for i, fm in enumerate(feature_maps_t):
        #     print(f"Teacher feature map {i} shape: {fm.shape}")

        # print("\n")
        # for i, fm in enumerate(feature_maps_s):
        #     print(f"Student feature map {i} shape: {fm.shape}")

    def backward(self, alpha=0.5, feature_weights=[0.1, 0.2, 0.3, 0.4]):
        """Compute losses and backpropagate"""
        self.optimizer.zero_grad()

        # Task-specific losses
        sr_loss_ts = self.criterion_wasserstein(
            self.student_outputs[0],
            self.teacher_outputs[0],
        )
        seg_ce_loss_ts = self.criterion_ce(
            self.student_outputs[1],
            self.teacher_outputs[1].argmax(dim=1),
        )
        seg_dice_loss_ts = self.criterion_dice_sm(
            self.student_outputs[1],
            self.teacher_outputs[1],
        )

        seg_loss_ts = seg_ce_loss_ts + seg_dice_loss_ts
        task_loss_ts = sr_loss_ts + seg_loss_ts

        # Task-specific losses for student
        sr_loss_gt = self.criterion_wasserstein(
            self.student_outputs[0],
            self.lr_images,
        )
        seg_ce_loss_gt = self.criterion_ce(
            self.student_outputs[1],
            self.labels,
        )
        seg_dice_loss_gt = self.criterion_dice_am(
            self.student_outputs[1],
            self.labels,
        )
        seg_loss_gt = seg_ce_loss_gt + seg_dice_loss_gt
        task_loss_gt = sr_loss_gt + seg_loss_gt

        task_loss = task_loss_ts + task_loss_gt

        # Feature distillation loss
        # if hasattr(self.teacher, 'feature_maps'):
        #     feature_loss = 0
        #     for student_fm, teacher_fm, weight in zip(
        #         self.student_feature_maps,
        #         self.teacher_feature_maps,
        #         feature_weights
        #     ):
        #         feature_loss += weight * self.criterion_wasserstein(
        #             student_fm,
        #             teacher_fm
        #         )
        # else:
        #     feature_loss = 0

        # # Embedding distillation loss
        # if hasattr(self.teacher, 'embedding'):
        #     embedding_loss = self.criterion_wasserstein(
        #         self.student_embedding,
        #         self.teacher_embedding
        #     )
        # else:
        #     embedding_loss = 0

        embedding_loss = self.criterion_wasserstein(
            self.student_embedding,
            self.teacher_embedding
        )

        feature_loss = 0.0

        # Combined loss
        total_loss = (1 - alpha) * task_loss + alpha * (0.5 * embedding_loss + 0.5 * feature_loss)

        total_loss.backward()

        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'sr_loss_ts': sr_loss_ts.item(),
            'seg_loss_ts': seg_loss_ts.item(),
            'sr_loss_gt': sr_loss_gt.item(),
            'seg_loss_gt': seg_loss_gt.item(),
            'feature_loss': feature_loss if isinstance(feature_loss, float) else feature_loss.item(),
            'embedding_loss': embedding_loss if isinstance(embedding_loss, float) else embedding_loss.item()
        }

    def train_step(self, data, alpha=0.5):
        """Perform one training step"""
        self.set_input(data)
        self.forward()
        return self.backward(alpha)

    def evaluate(self, val_loader):
        """Evaluate the model"""
        self.student.eval()
        total_sr_loss = 0
        total_seg_loss = 0
        total = 0

        with torch.no_grad():
            for data in val_loader:
                self.set_input(data)
                self.forward()

                # Compute SR and Segmentation losses
                sr_loss = self.criterion_wasserstein(
                    self.student_outputs[0],
                    self.teacher_outputs[0]
                )
                seg_loss = self.criterion_ce(
                    self.student_outputs[1],
                    self.teacher_outputs[1]
                )

                total_sr_loss += sr_loss.item()
                total_seg_loss += seg_loss.item()
                total += 1

        self.student.train()
        return {
            'sr_loss': total_sr_loss / total,
            'seg_loss': total_seg_loss / total,
            'total_loss': (total_sr_loss + total_seg_loss) / total
        }

    def save_checkpoint(self, path, epoch, metrics):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'student_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }, path)

    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.student.load_state_dict(checkpoint['student_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']


def trainer(model, train_loader, val_loader, num_epochs=10):
    for epoch in range(num_epochs):
        for data in train_loader:
            model.train_step(data)

def evaluator(model, val_loader):
    return model.evaluate(val_loader)
