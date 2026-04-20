# Satellite Landuse Segmentation

Repo nay hien co 2 luong chinh:
- huan luyen `ViTUnet` cho Sentinel-1 `2-band VV/VH` tu du lieu local trong `dataset/`
- infer va tinh dien tich tren bo crawl `inference_tif/Resolution3x3`

## Cau truc chinh
- `dataset/`: du lieu train local `*_sat.tif` + `*_mask.png`
- `inference_tif/Resolution3x3/`: TIFF dau vao de infer
- `inference_png/`: PNG output sau infer
- `weights/`: checkpoint cua moi run train
- `metrics/`: CSV/JSON metric cua moi run
- `images/`: sample prediction va confusion matrix
- `inference_model/`: checkpoint dung cho infer

## Moi truong
Dung env `landuse` va cai cac goi toi thieu neu chua co:

```bash
conda activate landuse
python -m pip install torch torchvision rasterio opencv-python einops python-dotenv tqdm matplotlib
```

## Train Sentinel-1 tu scratch
Script:
- [train_sentinel1_vitunet.sh](/mnt/disk1/aiotlab/anhnd2468/SatelliteLanduseSegmentation/train_sentinel1_vitunet.sh)

Lenh mac dinh:

```bash
conda activate landuse
cd /mnt/disk1/aiotlab/anhnd2468/SatelliteLanduseSegmentation
CUDA_VISIBLE_DEVICES=2 GPU_ID=0 BATCH_SIZE=2 EPOCHS=100 bash train_sentinel1_vitunet.sh
```

Bien moi truong quan trong:
- `GPU_ID`: index GPU ben trong danh sach da loc boi `CUDA_VISIBLE_DEVICES`
- `BATCH_SIZE`: batch size train
- `EPOCHS`: so epoch toi da
- `LR`: learning rate
- `DISABLE_WANDB=1`: tat wandb, mac dinh da tat
- `EARLY_STOP=1`: bat early stopping
- `PATIENCE=5`: so epoch patience

File output sau train:
- `weights/.../weight.pth`: best checkpoint tren GPU
- `weights/.../weight_cpu.pth`: best checkpoint CPU
- `metrics/.../train_metrics.csv`
- `metrics/.../best_val_metrics.json`
- `metrics/.../test_metrics.json`
- `metrics/.../run_summary.json`

## Train tiep tu best checkpoint hien tai
Script:
- [resume_sentinel1_vitunet.sh](/mnt/disk1/aiotlab/anhnd2468/SatelliteLanduseSegmentation/resume_sentinel1_vitunet.sh)

Mac dinh script nay:
- load best checkpoint Sentinel-1 hien tai
- LR = `1e-5`
- train toi da `15` epoch
- bat `early stopping`
- export checkpoint infer moi ra `inference_model/model_sentinel1_vitunet_resume_aug.pth`

Lenh:

```bash
conda activate landuse
cd /mnt/disk1/aiotlab/anhnd2468/SatelliteLanduseSegmentation
CUDA_VISIBLE_DEVICES=2 GPU_ID=0 BATCH_SIZE=2 bash resume_sentinel1_vitunet.sh
```

Neu muon chinh:

```bash
CUDA_VISIBLE_DEVICES=2 GPU_ID=0 BATCH_SIZE=2 EPOCHS=12 LR=5e-6 PATIENCE=4 bash resume_sentinel1_vitunet.sh
```

## Finetune Sentinel-1 tu checkpoint Sentinel-2 cu
Van dung script:
- [resume_sentinel1_vitunet.sh](/mnt/disk1/aiotlab/anhnd2468/SatelliteLanduseSegmentation/resume_sentinel1_vitunet.sh)

Khi dat `SOURCE=s2`, script se:
- load checkpoint `inference_model/model.pth`
- giu kien truc `ViTUnet` hien tai cho Sentinel-1
- chi bo qua layer dau vao khong khop shape `13 -> 2`
- export checkpoint moi ra `inference_model/model_sentinel1_vitunet_from_s2_finetune.pth`

Lenh:

```bash
conda activate landuse
cd /mnt/disk1/aiotlab/anhnd2468/SatelliteLanduseSegmentation
CUDA_VISIBLE_DEVICES=2 GPU_ID=0 SOURCE=s2 BATCH_SIZE=2 EPOCHS=15 LR=1e-5 PATIENCE=5 bash resume_sentinel1_vitunet.sh
```

Neu muon dat ten run va output rieng:

```bash
CUDA_VISIBLE_DEVICES=2 GPU_ID=0 SOURCE=s2 \
EXPERIMENT_NAME=sentinel1_vitunet_finetune_from_s2_ckpt_v2 \
EXPORT_PATH=/mnt/disk1/aiotlab/anhnd2468/SatelliteLanduseSegmentation/inference_model/model_sentinel1_vitunet_from_s2_finetune_v2.pth \
BATCH_SIZE=2 EPOCHS=15 LR=1e-5 PATIENCE=5 \
bash resume_sentinel1_vitunet.sh
```

## Infer bang checkpoint Sentinel-1 da train
Script:
- [infer_sentinel1.sh](/mnt/disk1/aiotlab/anhnd2468/SatelliteLanduseSegmentation/infer_sentinel1.sh)

Mac dinh:
- input: `inference_tif/Resolution3x3`
- output: `inference_png/Resolution3x3_sentinel1_vitunet_resume_aug`
- checkpoint: `inference_model/model_sentinel1_vitunet_resume_aug.pth`

Chay:

```bash
conda activate landuse
cd /mnt/disk1/aiotlab/anhnd2468/SatelliteLanduseSegmentation
CUDA_VISIBLE_DEVICES=2 GPU_ID=0 PATCH_SIZE=128 bash infer_sentinel1.sh
```

Neu muon chi dinh tay:

```bash
python infer.py \
  --cuda \
  --gpu_id 0 \
  --patch_size 128 \
  --model UNet \
  --input /mnt/disk1/aiotlab/anhnd2468/SatelliteLanduseSegmentation/inference_tif/Resolution3x3 \
  --output /mnt/disk1/aiotlab/anhnd2468/SatelliteLanduseSegmentation/inference_png/Resolution3x3_sentinel1_vitunet_resume_aug \
  --pretrained /mnt/disk1/aiotlab/anhnd2468/SatelliteLanduseSegmentation/inference_model/model_sentinel1_vitunet_resume_aug.pth
```

Output PNG se nam trong:
- `inference_png/Resolution3x3_sentinel1_vitunet_resume_aug`

## Tinh dien tich tu output infer Sentinel-1
Script:
- [calculate_area_sentinel1.sh](/mnt/disk1/aiotlab/anhnd2468/SatelliteLanduseSegmentation/calculate_area_sentinel1.sh)

Mac dinh script nay dung:
- input PNG: `inference_png/Resolution3x3_sentinel1_vitunet_resume_aug`
- output CSV: `area_output/Resolution3x3_sentinel1_vitunet_resume_aug`
- province label: `mapbox/gadm_resolution_3_province_mapbox_label.json`

Chay:

```bash
conda activate landuse
cd /mnt/disk1/aiotlab/anhnd2468/SatelliteLanduseSegmentation
bash calculate_area_sentinel1.sh
```

Neu muon goi truc tiep:

```bash
python calculate_area.py \
  --input /mnt/disk1/aiotlab/anhnd2468/SatelliteLanduseSegmentation/inference_png/Resolution3x3_sentinel1_vitunet_resume_aug \
  --output /mnt/disk1/aiotlab/anhnd2468/SatelliteLanduseSegmentation/area_output/Resolution3x3_sentinel1_vitunet_resume_aug \
  --province_label /mnt/disk1/aiotlab/anhnd2468/SatelliteLanduseSegmentation/mapbox/gadm_resolution_3_province_mapbox_label.json
```

File ket qua quan trong:
- `area_output/.../summary_province_area.csv`: tong dien tich tung lop theo tung `province_code`
- `area_output/.../per_image_province_area.csv`: chi tiet theo tile
- `area_output/.../summary_area.csv`: tong dien tich toan bo
- `area_output/.../duplicate_tiles.csv`: cac tile trung da bi loai khi cong dien tich

## Luong de dung nhat hien tai
1. Train hoac resume Sentinel-1.
2. Neu muon transfer tu checkpoint Sentinel-2 cu, chay `SOURCE=s2 bash resume_sentinel1_vitunet.sh`.
3. Lay checkpoint tot nhat trong `inference_model/model_sentinel1_vitunet_resume_aug.pth` hoac `inference_model/model_sentinel1_vitunet_from_s2_finetune.pth`.
4. Chay infer bang `bash infer_sentinel1.sh` hoac goi `infer.py` voi checkpoint can dung.
5. Chay tinh dien tich bang `bash calculate_area_sentinel1.sh`.

## Luu y
- `ViTUnet` Sentinel-1 hien tai dung input `512x512` trong train.
- `patch_size` infer nen de `128` hoac `256`, va phai la boi so cua `128`.
- `gpu_id` la chi so tuong doi trong danh sach da loc boi `CUDA_VISIBLE_DEVICES`.
- `infer.sh` cu va `inference_model/model.pth` cu thuoc luong checkpoint 13-channel truoc day; voi Sentinel-1 hien tai, uu tien dung `infer_sentinel1.sh`.
