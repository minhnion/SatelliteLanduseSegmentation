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

## Tinh dien tich

Mot entry point duy nhat la `calculate_area.py` voi 2 mode:
- `dataset`: tinh dien tich ground-truth tu `dataset/` (`*_sat.tif` + `*_mask.png`).
- `inference`: tinh dien tich output mo hinh tu `inference_tif/` (georef) ghep voi `inference_png/<run>/` (mask), tach theo 2 mua `Dong Xuan` (`0304_2023`) va `He Thu` (`0809_2023`).

Code nam trong package `area_calc/` (clean OOP, mỗi module 1 trach nhiem):
- `area_calc/config.py`: bang RGB <-> class, tag mua, hang so.
- `area_calc/geo.py`: `Province`, dien tich polygon (geodesic), dien tich pixel theo vi do, rasterize boundary.
- `area_calc/masks.py`: doc PNG, build mask theo tung class, dien tich tu mask.
- `area_calc/sources.py`: `DatasetSource`, `InferenceSource`, parse mua tu filename.
- `area_calc/calculator.py`: `AreaCalculator` xu ly mot tile -> `TileResult`.
- `area_calc/aggregator.py`: `SummaryAggregator` cong don, `ReportWriter` xuat CSV/JSON.
- `area_calc/cli.py`: argparse + dispatch.

Phuong phap (cho ca 2 mode):
- Doc transform/CRS/bounds tu TIF.
- Rasterize tung polygon tinh xuong grid pixel cua tile (`rasterio.features.rasterize`).
- Diem dien tich pixel theo `meters_per_degree` tai vi do trung tam moi row khi CRS la geographic, hoac `|a*e|` khi CRS projected.
- Dem pixel trong tung cap (province, class) tu mask, nhan voi dien tich pixel -> ra `m2`.

Can co boundary:
- `mapbox/vietnam_adm1_7provinces_2024.geojson`
- `mapbox/vietnam_adm1_7provinces_2024_metadata.json`

Neu chua co, tai bang:

```bash
python3 tools/download_vietnam_adm1_boundaries.py
```

### Dataset mode (ground truth QC)

```bash
conda activate landuse
cd /mnt/disk1/aiotlab/anhnd2468/SatelliteLanduseSegmentation
python calculate_area.py dataset \
  --dataset_dir dataset \
  --output area_output/dataset_ground_truth_gadm_boundary
```

Output:
- `summary_province_area.csv`: 7 dong, 1 dong/tinh — dien tich tung class + coverage.
- `per_image_province_area.csv`: 1 dong/(anh, tinh).
- `per_image_area.csv`: 1 dong/anh.
- `coverage_summary.csv`: tong coverage toan dataset.
- `missing_pairs.csv`, `method_metadata.json`.

### Inference mode (danh gia model AI)

```bash
conda activate landuse
cd /mnt/disk1/aiotlab/anhnd2468/SatelliteLanduseSegmentation
python calculate_area.py inference \
  --inference_tif_dir inference_tif/Resolution3x3 \
  --inference_png_dir inference_png/<thu_muc_run_model> \
  --output area_output/inference_by_boundary/<thu_muc_run_model>
```

Output:
- `summary_province_dong_xuan.csv`: 7 dong, dien tich Dong Xuan.
- `summary_province_he_thu.csv`: 7 dong, dien tich He Thu.
- `summary_province_by_season.csv`: 14 dong (7 tinh x 2 mua), tien pivot.
- `summary_province_combined.csv`: 7 dong, gop ca 2 mua.
- `per_image_province_area.csv`, `per_image_area.csv` co them cot `season_key`, `season_label`, `date_tag`.
- `coverage_summary.csv`: tong coverage cho moi mua.
- `missing_pairs.csv`, `method_metadata.json`.

### Cot quan trong trong summary

- `boundary_area_km2`: dien tich polygon tinh tu GADM.
- `covered_area_km2`: dien tich phan tile trung trong polygon tinh.
- `coverage_ratio_over_boundary`: `covered / boundary` (sat 1.0 la phu day).
- `uncovered_area_km2`: phan tinh chua co tile phu.
- `<class>_area_km2`, `<class>_area_ha`: dien tich tung class theo tinh.
- `unknown_color_*`: pixel co RGB khong nam trong bang class (canh bao mau la).

### Tuy chon them

- `--all_touched`: rasterize lay het pixel cham polygon (mac dinh chi pixel center).
- `--boundary` / `--boundary_metadata`: thay doi nguon boundary.

## Visualize dataset/ de QC label bang mat
Script:
- [visualize_dataset_qc.py](/mnt/disk1/aiotlab/anhnd2468/SatelliteLanduseSegmentation/visualize_dataset_qc.py)

Xuat top tile co ty le `Rice field` cao nhat:

```bash
conda activate landuse
cd /mnt/disk1/aiotlab/anhnd2468/SatelliteLanduseSegmentation
/mnt/disk1/aiotlab/envs/landuse/bin/python visualize_dataset_qc.py \
  --mode top-rice \
  --limit 50 \
  --output qc_visualization/dataset_top_rice_50
```

Xuat mot vai file cu the:

```bash
/mnt/disk1/aiotlab/envs/landuse/bin/python visualize_dataset_qc.py \
  --mode list \
  --files Area_2024_N_o_175 Area_2024_N_o_310 \
  --output qc_visualization/manual_check
```

Moi output gom:
- `sar_rgb/`: anh quicklook tu Sentinel-1 `VV/VH`
- `mask/`: mask label RGB goc
- `overlay/`: mask label phu len anh quicklook
- `index.csv`: duong dan output, toa do bbox, va ty le tung class trong tile

## Luong de dung nhat hien tai
1. Train hoac resume Sentinel-1.
2. Neu muon transfer tu checkpoint Sentinel-2 cu, chay `SOURCE=s2 bash resume_sentinel1_vitunet.sh`.
3. Lay checkpoint tot nhat trong `inference_model/model_sentinel1_vitunet_resume_aug.pth` hoac `inference_model/model_sentinel1_vitunet_from_s2_finetune.pth`.
4. Chay infer bang `bash infer_sentinel1.sh` hoac goi `infer.py` voi checkpoint can dung.
5. Chay tinh dien tich bang `python calculate_area.py inference --inference_png_dir inference_png/<run> --output area_output/inference_by_boundary/<run>`.

## Luu y
- `ViTUnet` Sentinel-1 hien tai dung input `512x512` trong train.
- `patch_size` infer nen de `128` hoac `256`, va phai la boi so cua `128`.
- `gpu_id` la chi so tuong doi trong danh sach da loc boi `CUDA_VISIBLE_DEVICES`.
- `infer.sh` cu va `inference_model/model.pth` cu thuoc luong checkpoint 13-channel truoc day; voi Sentinel-1 hien tai, uu tien dung `infer_sentinel1.sh`.
