import json
import os
import shutil
import traceback
import warnings
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from rasterio.errors import NotGeoreferencedWarning

from data.dataloader import load_dataloader, load_hr_dataloader
from dataset_config.load_config import load_config
from layers.unet_layers import *
from utils.evaluator import evaluate_on_test_set, evaluate_sr_seg_on_test_set, evaluate_scnet_on_test_set, evaluate_sr_on_test_set
from utils.model_utils import init_weights_he
from utils.parser import parse_args
from utils.seed import set_seed
from utils.trainer import train, train_sr_seg, train_scnet, train_sr_only

try:
    import wandb
except ImportError:
    wandb = None

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _slugify_tag(value):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))


def _save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def _resolve_input_size(model_name):
    # This ViT-UNet variant hard-codes a ViT bottleneck that expects a 32x32 feature map
    # after 4 downsampling stages, which corresponds to a 512x512 input image.
    if model_name in {"ViTUnet", "PromptedViTUnet"}:
        return 512
    return 1024


def _build_model(model_name, *, n_classes, n_channels, args, device, sr_cat, sr_output, scale_factor, h_w):
    if model_name == 'LinkNet':
        from model.LinkNet.model import LinkNet
        model = LinkNet(n_classes=n_classes, n_channels=n_channels)
    elif model_name == 'ViTUnet':
        from model.ViT.model import UNet
        model = UNet(n_classes=n_classes, n_channels=n_channels, depth=args.depth, heads=args.heads, dropout=args.dropout)
    elif model_name == 'PretrainedViT':
        from model.PretrainedViT.model import PretrainedViT
        model = PretrainedViT(n_classes=n_classes)
    elif model_name == 'PretrainedViTUNet':
        from model.PretrainedViTUNet.model import PretrainedViTUNet
        model = PretrainedViTUNet(n_channels=5, n_classes=n_classes)
    elif model_name == 'PromptedViTUnet':
        from model.PromptedVitUnet.model import PromptedVitUnet
        model = PromptedVitUnet(n_classes=n_classes, n_channels=n_channels, depth=args.depth, heads=args.heads, dropout=args.dropout)
    elif model_name == 'UNetSR':
        from model.UNetSR.model import UNetSR
        model = UNetSR(n_classes=n_classes, n_channels=n_channels, sr_cat=sr_cat, sr_output=sr_output)
    elif model_name == 'CrossAttentionUNetSR':
        from model.CrossAttentionUNetSR.model import CrossAttentionUNetSR
        model = CrossAttentionUNetSR(n_classes=n_classes, n_channels=n_channels)
    elif model_name == 'FCNResNet':
        from model.FCNResNet.model import FCNResnet
        model = FCNResnet(n_channels=n_channels, n_classes=n_classes)
    elif model_name == 'ESRT':
        from model.ESRT.model import ESRT
        model = ESRT(n_channels=3, n_classes=None, upscale=16)
    elif model_name == 'ESSRT':
        from model.ESRT.model import ESRT
        model = ESRT(n_channels=n_channels, n_classes=n_classes, upscale=scale_factor)
    elif model_name == 'FoundationModel':
        from model.Foundation.model import FoundationModel
        model = FoundationModel(n_channels=n_channels, n_classes=n_classes, upscale_factor=scale_factor)
    elif model_name == 'SCNet':
        from model.SCNet.model import SCNet
        model = SCNet(in_channels_Y=n_channels, in_channels_alpha=5 if n_channels == 13 else 4, n_classes=n_classes, scale_factor=scale_factor)
    elif model_name == 'LSKNet':
        from model.LSKNet.model import LSKNetSegmentor
        model = LSKNetSegmentor(n_classes=n_classes, n_channels=n_channels)
    elif model_name == 'PyramidMamba':
        from model.PyramidMamba.model import PyramidMamba
        model = PyramidMamba(n_classes=n_classes, n_channels=n_channels, img_size=h_w)
    elif model_name == 'FoundationKDModel':
        from model.FoundationKDNet.models import FoundationKDModel
        model = FoundationKDModel(n_channels=n_channels, n_classes=n_classes, upscale_factor=scale_factor)
    elif model_name == 'FoundationKDModelHR':
        from model.FoundationKDNet.models import FoundationKDModel
        model = FoundationKDModel(n_channels=n_channels, n_classes=n_classes, upscale_factor=16, sr_out_chan=3)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model.to(device)

if __name__ == "__main__":
    sr_seg_models = ['ESSRT', 'FoundationModel', 'FoundationKDModel', 'FoundationKDModelHR', 'UNetSR', 'CrossAttentionUNetSR', 'SCNet', 'LSKNet', 'PyramidMamba']
    sr_only_models = ['ESRT']
    try:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        # Load environment variables from .env file
        load_dotenv()

        # Initialize
        set_seed(20)
        args = parse_args()
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        use_cuda = args.cuda and torch.cuda.is_available()
        if args.cuda and not torch.cuda.is_available():
            print("CUDA requested but not available, using CPU instead.")
        if use_cuda:
            visible_device_count = torch.cuda.device_count()
            if args.gpu_id < 0 or args.gpu_id >= visible_device_count:
                raise ValueError(
                    f"Requested gpu_id={args.gpu_id}, but PyTorch sees {visible_device_count} visible CUDA device(s). "
                    f"CUDA_VISIBLE_DEVICES={cuda_visible_devices!r}. If CUDA_VISIBLE_DEVICES is set, gpu_id is relative "
                    f"to that filtered list. Example: CUDA_VISIBLE_DEVICES=2 with --gpu_id 0 targets physical GPU 2."
                )
        device = torch.device("cuda:" + str(args.gpu_id)) if use_cuda else torch.device("cpu")
        print(f"Using device: {device}")

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
        dataset_label = _slugify_tag(os.path.basename(os.path.abspath(dataset_dir)))
        run_label = _slugify_tag(args.experiment_name) if args.experiment_name else dataset_label

        model_log_path = (
            f'datasetcfg:{dataset}/data:{dataset_label}/channels:{n_channels}/model:{args.model}/run:{run_label}/'
            f'datetime:{timestamp}_epoch:{args.epoch}_bs:{args.batch_size}_lr:{args.lr}/'
        )
        weight_path = os.path.join(log_path, 'weights', model_log_path)
        image_path = os.path.join(log_path, 'images', model_log_path)
        metrics_path = os.path.join(log_path, 'metrics', model_log_path)

        if not os.path.exists(weight_path):
            os.makedirs(weight_path)
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        if not os.path.exists(metrics_path):
            os.makedirs(metrics_path)

        weight_file_path = os.path.join(weight_path, 'weight.pth')
        train_metrics_path = os.path.join(metrics_path, 'train_metrics.csv')
        best_metrics_path = os.path.join(metrics_path, 'best_val_metrics.json')
        test_metrics_path = os.path.join(metrics_path, 'test_metrics.json')

        print(f"Dataset dir: {dataset_dir}")
        print(f"Weight dir: {weight_path}")
        print(f"Image dir: {image_path}")
        print(f"Metrics dir: {metrics_path}")

        model = None
        sr_output = True
        sr_cat = False
        scale_factor = 2
        mask_scale = None

        h_w = _resolve_input_size(args.model)
        size = (h_w, h_w)

        model = _build_model(
            args.model,
            n_classes=n_classes,
            n_channels=n_channels,
            args=args,
            device=device,
            sr_cat=sr_cat,
            sr_output=sr_output,
            scale_factor=scale_factor,
            h_w=h_w,
        )
        if args.model == 'UNetSR':
            sr_output = True
            mask_scale = scale_factor if sr_output else None

        if args.pretrained:
            checkpoint = args.pretrained
            pretrained_state = torch.load(checkpoint, map_location="cpu")
            if isinstance(pretrained_state, dict) and isinstance(pretrained_state.get("state_dict"), dict):
                pretrained_state = pretrained_state["state_dict"]
            if not isinstance(pretrained_state, dict):
                raise ValueError(f"Unsupported checkpoint format: {checkpoint}")

            model_dict = model.state_dict()
            matched_layers = {}
            mismatched_layers = []
            unexpected_layers = []

            for key, value in pretrained_state.items():
                if key not in model_dict:
                    unexpected_layers.append(key)
                    continue
                if not torch.is_tensor(value):
                    unexpected_layers.append(key)
                    continue
                if value.size() == model_dict[key].size():
                    matched_layers[key] = value
                else:
                    mismatched_layers.append((key, tuple(value.size()), tuple(model_dict[key].size())))

            model_dict.update(matched_layers)
            model.load_state_dict(model_dict)

            missing_layers = [key for key in model_dict if key not in matched_layers]
            print(
                f"Loaded pretrained checkpoint: matched {len(matched_layers)}/{len(model_dict)} tensors, "
                f"mismatched {len(mismatched_layers)}, unexpected {len(unexpected_layers)}, missing {len(missing_layers)}"
            )
            if mismatched_layers:
                print("Mismatched pretrained tensors:")
                for key, source_shape, target_shape in mismatched_layers[:10]:
                    print(f"  {key}: checkpoint {source_shape} -> model {target_shape}")
                if len(mismatched_layers) > 10:
                    print(f"  ... {len(mismatched_layers) - 10} more")
        # else:
        #     model.apply(init_weights_he)

        model = model.to(device)

        print("model loaded")
        if args.model =='FoundationKDModelHR':
            train_loader, val_loader, test_loader = load_hr_dataloader(batch_size=args.batch_size, classes=classes, RGB_TO_CLASSES=RGB_TO_CLASSES, root_dir=dataset_dir, size=size, num_tiles=num_tiles, scale_factor=16)
        elif args.model in sr_seg_models or args.model in sr_only_models:
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
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        torch.cuda.empty_cache()

        wandb_enabled = False
        run = None
        if args.disable_wandb:
            print("W&B logging disabled by flag")
        elif wandb is None:
            print("W&B disabled because the package is not installed")
        else:
            try:
                WANDB_API_KEY = os.getenv('WANDB_API_KEY')
                if WANDB_API_KEY:
                    wandb.login(key=WANDB_API_KEY)
                else:
                    wandb.login()
                run = wandb.init(
                    entity='mingixpt-hust',
                    project=os.path.basename(dataset_dir),
                    config=CFG,
                    save_code=True,
                    job_type='train',
                    group=f'{args.model}',
                    name=f'model:{args.model}_run:{run_label}_epoch:{args.epoch}_bs:{args.batch_size}_lr:{args.lr}_datetime:{timestamp}'
                )
                wandb_enabled = run is not None
            except Exception as wandb_error:
                print(f"W&B disabled because initialization failed: {wandb_error}")

        num_epochs = 1 if base_path == 'dummy_dataset' else args.epoch

        run_summary = {
            "timestamp": timestamp,
            "device": str(device),
            "dataset_config": dataset,
            "dataset_dir": os.path.abspath(dataset_dir),
            "dataset_base_path_config": base_path,
            "n_channels": n_channels,
            "n_classes": n_classes,
            "classes": classes,
            "model": args.model,
            "depth": args.depth,
            "heads": args.heads,
            "dropout": args.dropout,
            "input_size": h_w,
            "epochs": num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "pretrained": args.pretrained,
            "early_stop": args.early_stop,
            "patience": args.patience,
            "wandb_enabled": wandb_enabled,
            "run_label": run_label,
            "weight_dir": os.path.abspath(weight_path),
            "image_dir": os.path.abspath(image_path),
            "metrics_dir": os.path.abspath(metrics_path),
            "best_checkpoint_path": os.path.abspath(weight_file_path),
            "best_checkpoint_cpu_path": os.path.abspath(weight_file_path.replace(".pth", "_cpu.pth")),
            "train_metrics_path": os.path.abspath(train_metrics_path),
            "best_metrics_path": os.path.abspath(best_metrics_path),
            "test_metrics_path": os.path.abspath(test_metrics_path),
            "train_dataset_len": len(train_loader.dataset),
            "val_dataset_len": len(val_loader.dataset),
            "test_dataset_len": len(test_loader.dataset),
            "train_batches": len(train_loader),
            "val_batches": len(val_loader),
            "test_batches": len(test_loader),
            "split_strategy": "existing_train_val_test_dirs" if all(
                os.path.exists(os.path.join(dataset_dir, split_name)) for split_name in ("train", "val", "test")
            ) else "auto_random_split_70_20_10",
        }
        if args.export_inference_path:
            run_summary["export_inference_path"] = os.path.abspath(args.export_inference_path)
        _save_json(os.path.join(metrics_path, "run_summary.json"), run_summary)

        if args.model in sr_seg_models:
            model = train_sr_seg(model, train_loader, val_loader, optimizer, scheduler, criterion_seg, criterion_sr, classes, device, l1_lambda=0, num_epochs=num_epochs, save_path=weight_file_path, image_dir=image_path, early_stop=args.early_stop, patience=args.patience)
            evaluate_sr_seg_on_test_set(model, test_loader, classes, CLASSES_TO_RGB, image_dir=image_path, wandb_setup=wandb_enabled, num_samples=1)
        elif args.model == 'SCNet':
            model = train_scnet(model, train_loader, val_loader, optimizer, scheduler, criterion_seg, classes, device, l1_lambda=0, num_epochs=num_epochs, save_path=weight_file_path, image_dir=image_path, early_stop=args.early_stop, patience=args.patience)
            evaluate_scnet_on_test_set(model, test_loader, classes, CLASSES_TO_RGB, image_dir=image_path, wandb_setup=wandb_enabled, num_samples=1)
        elif args.model in sr_only_models:
            model = train_sr_only(model, train_loader, val_loader, optimizer, scheduler, criterion_sr, classes, device, l1_lambda=0, num_epochs=num_epochs, save_path=weight_file_path, image_dir=image_path, early_stop=args.early_stop, patience=args.patience)
            evaluate_sr_on_test_set(model, test_loader, classes, CLASSES_TO_RGB, image_dir=image_path, wandb_setup=wandb_enabled, num_samples=1)
        else:
            model = train(
                model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                criterion_seg,
                classes,
                device,
                l1_lambda=0,
                num_epochs=num_epochs,
                save_path=weight_file_path,
                image_dir=image_path,
                early_stop=args.early_stop,
                patience=args.patience,
                train_metrics_path=train_metrics_path,
                best_metrics_path=best_metrics_path,
                wandb_setup=wandb_enabled,
            )
            evaluate_on_test_set(
                model,
                test_loader,
                classes,
                CLASSES_TO_RGB,
                image_dir=image_path,
                wandb_setup=wandb_enabled,
                num_samples=1,
                metrics_path=test_metrics_path,
            )

        if args.export_inference_path:
            exported_checkpoint_path = args.export_inference_path
            export_dir = os.path.dirname(exported_checkpoint_path)
            if export_dir:
                os.makedirs(export_dir, exist_ok=True)
            source_checkpoint = weight_file_path.replace('.pth', '_cpu.pth')
            if not os.path.exists(source_checkpoint):
                source_checkpoint = weight_file_path
            shutil.copy2(source_checkpoint, exported_checkpoint_path)
            print(f"Exported inference checkpoint to: {exported_checkpoint_path}")
            run_summary["exported_inference_checkpoint_path"] = os.path.abspath(exported_checkpoint_path)

        _save_json(os.path.join(metrics_path, "run_summary.json"), run_summary)

        if run is not None:
            run.finish()

    except Exception as e:
        print("An error occurred:", str(e))
        traceback.print_exc()
