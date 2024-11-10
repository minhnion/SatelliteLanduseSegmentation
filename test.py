from model.PromptedVitUnet.model import PromptedVitUnet
from utils.image_utils import open_tif_image

if __name__ =="__main__":
    # classes = ['unidentifiable', 'forest', 'rice_field', 'water', 'residential']
    # device = torch.device("cuda:0" ) 
    # data = torch.FloatTensor(8, 5, 256, 256).to(device)

    # model = PromptedVitUnet(n_classes=len(classes), n_channels=5, depth=8, heads=8).to(device)
    # print(model(data))

    img = open_tif_image("/mnt/henryng/sentinel_dataset_cut256_filtered/train/anh_25km_31_39_sat.tif")
    print(img)