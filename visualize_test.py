import torch

from data.dataloader import load_dataloader
from model.PromptedVitUnet.model import PromptedVitUnet

def visualize_test(model, test_dir):
    pass 
    
if __name__=="__main__":
    device = torch.device("cuda:" + str('0'))
    classes = ['unidentifiable', 'forest', 'rice_field', 'water', 'residential']
    root_dir = '/mnt/henryng/augment_images_dataset/augment_images_dataset'
    n_channels = 5
    n_classes = len(classes)
    
    _, _ , test_loader = load_dataloader(batch_size=32, root_dir=root_dir)
    
    model = PromptedVitUnet(n_classes=n_classes, n_channels=n_channels).to(device)