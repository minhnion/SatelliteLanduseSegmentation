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
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument("--gpu_id", type=int, default=1, help="gpu id")
    parser.add_argument('--dim', type=int, default=128, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-3, help='l2 regularization weight')
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs')
    parser.add_argument('--depth', type=int, default=12, help='depth')
    parser.add_argument('--heads', type=int, default=12, help='number of heads')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--pretrained', type=str, default=None, help='use pretrained model')
    parser.add_argument('--dataset', type=str, default='north_vn', help='dataset')
    parser.add_argument('--data_path', type=str, default='.', help='data set path')
    parser.add_argument('--log_path', type=str, default='.', help='logging path')
    parser.set_defaults(feature=True)

    return parser.parse_args()

def parse_infer_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cuda", action='store_true', help='Use CUDA if available')
    parser.add_argument("--input", type=str, help='input image path')
    parser.add_argument("--output", type=str, help='output image path')
    parser.add_argument("--pretrained", type=str, help='pretrained model path')
    parser.add_argument("--patch_size", type=int, default=256, help='patch size')
    parser.add_argument("--model", type=str, default='FoundationModel', help='model to use')
    return parser.parse_args()
