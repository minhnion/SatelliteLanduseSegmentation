import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cuda", action='store_true', help='Use CUDA if available')
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            'ViTUnet',
            'PretrainedViT',
            'LinkNet',
            'PromptedViTUnet',
            'Segformer',
            'PretrainedViTUNet',
            'UNetSR',
            'CrossAttentionUNetSR',
            'FCNResNet',
            'ESRT',
            'ESSRT',
            'FoundationModel',
            'SCNet',
            'LSKNet',
            'PyramidMamba',
            'FoundationKDModel',
            'FoundationKDModelHR',
        ],
        default='ViTUnet',
        help="model to train"
    )
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-3, help='l2 regularization weight')
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker count')
    parser.add_argument('--depth', type=int, default=8, help='depth')
    parser.add_argument('--heads', type=int, default=8, help='number of heads')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--pretrained', type=str, default=None, help='use pretrained model')
    parser.add_argument(
        '--resume_checkpoint',
        type=str,
        default=None,
        help='resume a full training-state checkpoint with optimizer/scheduler/epoch state',
    )
    parser.add_argument('--dataset', type=str, default='north_vn', help='dataset')
    parser.add_argument('--data_path', type=str, default='dataset', help='data set path')
    parser.add_argument('--log_path', type=str, default='.', help='logging path')
    parser.add_argument('--run_dir', type=str, default=None, help='explicit run directory containing checkpoints/images/metrics')
    parser.add_argument('--experiment_name', type=str, default=None, help='optional readable run name')
    parser.add_argument('--export_inference_path', type=str, default=None, help='optional path to export best checkpoint for inference')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'AdamW'], help='optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay')
    parser.add_argument(
        '--monitor_metric',
        type=str,
        default='val_loss',
        choices=['val_loss', 'val_miou'],
        help='metric used for best checkpoint and early stopping',
    )
    parser.add_argument('--save_every', type=int, default=0, help='save epoch training-state every N epochs; 0 disables')
    parser.add_argument('--tf32', action='store_true', help='allow TF32 CUDA kernels during training')
    parser.add_argument('--save_full_snapshots', action='store_true', help='also save full pickled model snapshots on each new best')
    parser.add_argument('--disable_wandb', action='store_true', help='disable Weights & Biases logging')
    parser.add_argument('--early_stop', action='store_true', help='enable early stopping')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience in epochs')
    parser.set_defaults(feature=True)

    return parser.parse_args()

def parse_infer_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cuda", action='store_true', help='Use CUDA if available')
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument("--input", type=str, default='inference_tif', help='input GeoTIFF file or folder')
    parser.add_argument("--output", type=str, default='inference_png/sentinel1_best', help='output image folder')
    parser.add_argument("--pretrained", type=str, default='inference_model/model_sentinel1_best.pth', help='pretrained model path')
    parser.add_argument("--patch_size", type=int, default=512, help='raw Sentinel-1 tile/patch size before model resize; 512 runs one full-tile pass for current 3km S1 tiles')
    parser.add_argument("--stride", type=int, default=None, help='raw patch stride; default is patch_size // 2')
    parser.add_argument("--model_input_size", type=int, default=512, help='ViTUnet input size used during training')
    parser.add_argument("--model", type=str, default='UNet', choices=['UNet', 'FoundationModel'], help='model to use')
    parser.add_argument("--no_recursive", action='store_true', help='do not search input folders recursively')
    parser.add_argument("--preserve_dirs", action='store_true', help='preserve input subfolders in output folder')
    parser.add_argument("--limit", type=int, default=None, help='optional max number of TIF files to process')
    parser.add_argument("--dry_run", action='store_true', help='list matched files without running model inference')
    parser.add_argument("--fp16", action='store_true', help='use FP16 inference on CUDA; faster/lower memory but less numerically precise')
    parser.add_argument("--tf32", action='store_true', help='allow TF32 CUDA kernels; faster on Ampere/A100 but slightly less precise than full FP32')
    parser.add_argument(
        "--allow_sentinel1_to_13_adapter",
        action='store_true',
        help='allow the legacy pseudo-13-band adapter when a 13-channel checkpoint is used on 2-band Sentinel-1 input',
    )
    return parser.parse_args()
