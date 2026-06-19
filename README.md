# Satellite Landuse Segmentation

Dự án phân vùng sử dụng đất từ ảnh vệ tinh. Luồng chính hiện tại là Sentinel-1 với kiến trúc `ViTUnet`, finetune từ checkpoint Sentinel-2 tốt nhất `inference_model/model.pth`.

## Môi Trường

```bash
conda activate landuse
pip install -r requirements.txt
```

`requirements.txt` phục vụ luồng train/finetune chính `ViTUnet`. Nếu train
model thử nghiệm `PyramidMamba`, cài thêm:

```bash
pip install -r requirements-pyramidmamba.txt --no-build-isolation
```

Lưu ý `mamba_ssm/flash_attn` cần `nvcc` tương thích với CUDA của PyTorch. Với
`torch==2.11.0+cu128`, không build bằng compiler CUDA 11.x.

Nếu chọn GPU cụ thể:

```bash
export CUDA_VISIBLE_DEVICES=0
export GPU_ID=0
```

`GPU_ID` là index GPU mà PyTorch nhìn thấy sau khi lọc bằng `CUDA_VISIBLE_DEVICES`.

## Checkpoint CPU Và GPU

`best_model_cpu.pth` không có nghĩa là infer chạy CPU. Nó chỉ là checkpoint được lưu với tensor nằm trên CPU để dễ load trên mọi máy.

Khi infer/train có `--cuda`, model vẫn chạy GPU sau khi checkpoint được load vào model và model được `.to(cuda)`.

Khuyến nghị dùng để copy sang `inference_model/`:

```bash
cp <run_dir>/checkpoints/best_model_cpu.pth inference_model/model_sentinel1_best.pth
```

`best_model.pth` cũng dùng được, nhưng `best_model_cpu.pth` portable và an toàn hơn.

## Finetune Sentinel-1 Từ Sentinel-2 Best

Run mặc định chất lượng cao:

```bash
bash finetune_sentinel1_vitunet_from_s2_best.sh
```

Output nằm trong:

```text
finetune_runs/sentinel1_vitunet_from_s2/<run_name_timestamp>/
```

File quan trọng:

```text
checkpoints/best_model_cpu.pth
checkpoints/best_model.pth
checkpoints/latest_training_state.pth
checkpoints/best_training_state.pth
metrics/train_metrics.csv
metrics/best_val_metrics.json
metrics/test_metrics.json
metrics/run_summary.json
```

Tuỳ chỉnh nhanh:

```bash
RUN_NAME=s1_vitunet_from_s2_v2 EPOCHS=150 BATCH_SIZE=4 LR=0.00005 bash finetune_sentinel1_vitunet_from_s2_best.sh
```

## Train Sentinel-1 From Scratch

Dùng để làm baseline so sánh với finetune từ Sentinel-2:

```bash
PRETRAINED=none RUN_NAME=s1_vitunet_scratch bash finetune_sentinel1_vitunet_from_s2_best.sh
```

## Continue Và Resume

`<previous_run_dir>` hoặc `<run_dir>` là placeholder, phải thay bằng thư mục run thật. Không gõ dấu `< >`.

Continue từ best weight của run cũ: tạo run mới, load `best_model_cpu.pth` của run cũ, optimizer/scheduler bắt đầu lại. Dùng khi run cũ đã xong và muốn train thêm phase mới.

```bash
bash continue_sentinel1_vitunet_from_run_best.sh finetune_runs/sentinel1_vitunet_from_s2/s1_vitunet_from_s2_quality_20260607_120000
```

Ví dụ đổi tên run mới và LR nhỏ hơn:

```bash
RUN_NAME=s1_continue_lr1e5 LR=0.00001 EPOCHS=50 bash continue_sentinel1_vitunet_from_run_best.sh finetune_runs/sentinel1_vitunet_from_s2/s1_vitunet_from_s2_quality_20260607_120000
```

Resume run bị dừng giữa chừng: tiếp tục đúng run cũ, restore cả model, optimizer, scheduler, epoch và early-stop counter từ `latest_training_state.pth`.

```bash
RESUME_CHECKPOINT=finetune_runs/sentinel1_vitunet_from_s2/s1_vitunet_from_s2_quality_20260607_120000/checkpoints/latest_training_state.pth RUN_DIR=finetune_runs/sentinel1_vitunet_from_s2/s1_vitunet_from_s2_quality_20260607_120000 bash finetune_sentinel1_vitunet_from_s2_best.sh
```

## TF32 Và FP16

Mặc định training/infer dùng FP32 chính xác nhất.

Bật TF32 để nhanh hơn trên A100/Ampere, đổi lại kém chính xác số học hơn một chút:

```bash
TF32=1 bash finetune_sentinel1_vitunet_from_s2_best.sh
```

Infer FP16 chỉ dùng khi muốn nhanh/nhẹ bộ nhớ hơn:

```bash
FP16=1 bash infer_sentinel1.sh
```

## Infer Sentinel-1

Đặt checkpoint đã chọn vào `inference_model/`, ví dụ:

```bash
cp <run_dir>/checkpoints/best_model_cpu.pth inference_model/model_sentinel1_best.pth
```

Chạy infer toàn bộ `inference_tif/`:

```bash
INPUT=inference_tif OUTPUT=inference_png/sentinel1_best PRETRAINED_MODEL=inference_model/model_sentinel1_best.pth bash infer_sentinel1.sh
```

Mặc định inference dùng raw patch `140x140`, stride `70`, rồi resize từng
patch lên `512x512`. Cấu hình này khớp với lúc train: ảnh nguồn khoảng
`558x558` được chia thành lưới `4x4` trước khi resize. Không dùng
`PATCH_SIZE=512` cho checkpoint này vì nó thay đổi mạnh phạm vi không gian mà
model nhìn thấy.

Kiểm tra danh sách file trước khi chạy model:

```bash
DRY_RUN=1 INPUT=inference_tif bash infer_sentinel1.sh
```

Giữ cấu trúc thư mục con output:

```bash
PRESERVE_DIRS=1 INPUT=inference_tif OUTPUT=inference_png/sentinel1_best bash infer_sentinel1.sh
```

## Tính Diện Tích

Tính diện tích từ label dataset gốc:

```bash
python calculate_area.py dataset --dataset_dir dataset --output area_output/dataset_ground_truth
```

Tính diện tích từ kết quả infer:

```bash
python calculate_area.py inference --inference_tif_dir inference_tif --inference_png_dir inference_png/sentinel1_best --output area_output/inference_sentinel1_best
```

Output chính:

```text
summary_province_by_season.csv
summary_province_dong_xuan.csv
summary_province_he_thu.csv
summary_province_combined.csv
per_image_province_area.csv
coverage_summary.csv
method_metadata.json
```

## Notebook QC

Dataset label audit:

```text
notebooks/dataset_label_audit.ipynb
```

Inference visual audit:

```text
notebooks/inference_visual_audit.ipynb
```
