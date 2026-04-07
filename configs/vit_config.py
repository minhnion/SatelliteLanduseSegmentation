from mmcv import Config

from mmseg.apis import set_random_seed
from mmseg.utils import get_device

def get_config(config_path, num_classes, data_root, batch_size, epochs, lr, gpu_id):
    cfg = Config.fromfile(config_path)
    cfg.norm_cfg = dict(type='BN', requires_grad=True)
    cfg.model.backbone.norm_cfg = cfg.norm_cfg
    cfg.model.decode_head.norm_cfg = cfg.norm_cfg
    cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg

    cfg.model.decode_head.num_classes = num_classes
    cfg.model.auxiliary_head.num_classes = num_classes

    cfg.dataset_type = 'LandCoverDataset'
    cfg.data_root = data_root

    cfg.data.samples_per_gpu = batch_size
    cfg.data.workers_per_gpu = 4
    cfg.crop_size = (512, 512)

    cfg.data.tran.type = cfg.dataset_type
    cfg.data.train.data_root = cfg.data_root
    cfg.data.train.img_dir = data_root + '/images'
    cfg.data.train.ann_dir = data_root + '/masks'
    
