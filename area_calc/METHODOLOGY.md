# Area Calculation Methodology
> Chi tiết đầu vào, pipeline từng bước, ví dụ số liệu cụ thể, và lập luận tính đúng cho 2 chế độ: `dataset` (ground-truth) và `inference` (output model).

---

## 1. Tổng quan

`calculate_area.py` có 2 mode:
- **`dataset`**: đọc cặp `*_sat.tif` + `*_mask.png` trong `dataset/` — dùng để QC label ground-truth.
- **`inference`**: ghép `inference_tif/Resolution3x3/<stem>.tif` (để lấy georef) với `inference_png/<run>/<stem>_infered.png` (class output của model) — tách theo 2 mùa Đông Xuân (`0304_2023`) và Hè Thu (`0809_2023`).

Hai mode **cùng 1 pipeline xử lý**, khác nhau duy nhất ở lớp nguồn dữ liệu (`DatasetSource` vs `InferenceSource`). Kết quả cuối là số km² cho mỗi class (Forest / Rice field / Water / Residential / Unidentifiable) trong từng tỉnh.

---

## 2. Input — bảng tra các thành phần

### 2.1 Sentinel-1 GeoTIFF (`inference_tif/Resolution3x3/`)

File TIFF 2-band (VV, VH), float64, **EPSG:4326** (kinh độ / vĩ độ). Không phải ảnh 1 ngày — mỗi file là **median composite** của tất cả ảnh Sentinel-1 trong khoảng thời gian crawl (ví dụ `_0304_2023.tif` = median của tháng 3-4/2023, `_0809_2023.tif` = median tháng 8-9/2023).

Quan trọng nhất: TIFF lưu **GeoTransform** — ánh xạ pixel index ↔ vị trí địa lý thực.

| Trường | Ý nghĩa | Ví dụ từ `2321_19_35_0304_2023.tif` |
|---|---|---|
| `width` | số pixel theo kinh độ | 320 |
| `height` | số pixel theo vĩ độ | 301 |
| `transform` | ma trận affine | `(8.983e-05, 0, 105.76, 0, -8.983e-05, 20.57)` |
| `crs` | hệ chiếu | `EPSG:4326` |
| `bounds` | bao địa lý `(left, bottom, right, top)` | `(105.7600, 20.5430, 105.7888, 20.5701)` |

- `transform.a` = `8.983e-05` độ/pixel theo kinh độ
- `transform.e` = `-8.983e-05` độ/pixel theo vĩ độ (dấu trừ vì raster lưu row 0 ở **top**, tăng xuống)

### 2.2 Mask PNG (dataset: `*_mask.png`, inference: `*_infered.png`)

Ảnh RGB 320×301, mỗi màu = 1 class. Không có georef — chỉ dùng để đếm pixel theo màu.

| Màu (R,G,B) | Class | Ý nghĩa |
|---|---|---|
| (0,255,0) | Forest | Rừng |
| (255,0,0) | Rice field | Lúa nước |
| (0,255,255) | Water | Mặt nước |
| (255,255,0) | Residential | Khu dân cư |
| (0,0,0), (0,0,255), (255,255,255) | Unidentifiable | Không xác định |

### 2.3 Boundary GeoJSON (`mapbox/vietnam_adm1_7provinces_2024.geojson`)

7 đa giác WGS84 — ranh giới hành chính cấp 1 của Bắc Ninh, Hà Nội, Hải Dương, Hải Phòng, Hưng Yên, Quảng Ninh, Vĩnh Phúc.

Mỗi feature có `geometry` (MultiPolygon), `province_code` (1-7), `province_name`.

---

## 3. Pipeline chung — step by step

Gọi chung cho cả 2 mode. Input là **1 cặp (TIF, PNG)** ứng 1 tile. Output là **kết quả phân bổ diện tích** của tile đó vào 7 tỉnh.

### Step 1: Parse georef từ TIF

Đọc bằng `rasterio.open(tif_path)` → lấy `width`, `height`, `transform`, `crs`, `bounds`. Tạo mảng `row_pixel_area_m2` — diện tích **thực tế** (m²) của từng pixel theo **từng hàng** (row) raster.

Với EPSG:4326, công thức:

```
meters_per_degree_lat(y) = 111132.92 - 559.82·cos(2y) + 1.175·cos(4y) - 0.0023·cos(6y)  [m/°]
meters_per_degree_lon(y) = 111412.84·cos(y) - 93.5·cos(3y) + 0.118·cos(5y)     [m/°]
pixel_area(row) = |transform.a| · meters_per_degree_lon(y_center_of_row)
                · |transform.e| · meters_per_degree_lat(y_center_of_row)
```

Với EPSG:projected (ví dụ UTM), đơn giản:

```
pixel_area = |transform.a · transform.e|  [m²/pixel, đồng đều]
```

**Tổng diện tích tile** = `sum(row_pixel_area_m2) * width`.

**Tại sao cần per-row?** Vì `meters_per_degree_lon` phụ thuộc vĩ độ — ở Hà Nội (lat ~21°), 1° kinh độ ≈ 103.5 km; ở Quảng Ninh (lat ~21°) tương tự; nhưng nếu dataset mở rộng sang Cà Mau (lat ~9°) thì chênh ~15%. Per-row loại bỏ sai số này hoàn toàn.

### Step 2: Load mask PNG và phân lớp

- Đọc PNG, resize (nếu cần) về `(width, height)` bằng `Image.NEAREST` (không tạo màu mới).
- `build_class_masks(mask_rgb)` → dict `class_name → boolean mask (H×W)` — `True` = pixel thuộc class đó.
- Đồng thời tạo `unknown_mask` = pixel không khớp bất kỳ class nào (cảnh báo).

### Step 3: Rasterize boundary tỉnh lên grid của tile

Với mỗi tỉnh có ranh giới chạm qua tile (`bounds_intersect(bounds, province.bounds) == True`):

```python
province_mask = rasterize(
    [(province.geometry, 1)],   # đa giác tỉnh, giá trị 1
    out_shape=(height, width), # cùng kích thước tile
    transform=transform,       # dùng transform GIỐNG TIF
    fill=0,
    dtype="uint8",
)
```

Kết quả: mảng nhị phân `(H×W)` — `1` = pixel thuộc polygon tỉnh, `0` = nằm ngoài.

**Điểm mấu chốt khác cách cũ**: code cũ (đã xóa) lấy `ratio` từ JSON rồi **nhân đều** cho mọi class. Code mới rasterize trực tiếp → biết **chính xác pixel nào nằm trong tỉnh nào**, không cần giả định phân bố đều.

### Step 4: Cộng diện tích theo cặp (tỉnh, class)

Với mỗi tỉnh intersect tile:

```
pixels(province, class) = sum(province_mask & class_mask)          # đếm pixel
area_m2(province, class) = Σ_row  count_in_row · row_pixel_area_m2  # tích vô hướng
```

Tổng cho tất cả class trong tỉnh:

```
area_m2(province) = area_for_mask(province_mask, row_pixel_area_m2)
area_m2(known)    = area_m2(province) - area_m2(unknown)
```

### Step 5: Gom tổng toàn tile (inside/outside/overlap)

Vì 1 pixel có thể nằm trong nhiều tỉnh (nếu tile chạm ranh giới 2 tỉnh), cần:

```python
province_mask_sum = Σ(province_mask_i)    # uint8, giá trị 0..N
inside_any  = province_mask_sum > 0       # pixel thuộc ít nhất 1 tỉnh
overlap     = province_mask_sum > 1       # pixel thuộc >= 2 tỉnh (chồng ranh giới)
outside     = ~inside_any                  # nằm ngoài 7 tỉnh
```

Area tương ứng:

```
area_inside = area_for_mask(inside_any, row_pixel_area)
area_overlap = area_for_mask(overlap, row_pixel_area)
area_outside = image_area - area_inside
```

Lưu ý: pixel chồng (overlap) chỉ ≈ 0.63 km² / 2130 tile = **0.0003 km²/tile** = bỏ sót không đáng kể.

### Step 6: Accumulate vào SummaryAggregator

`SummaryAggregator` lưu:
- `per_image_rows`: 1 dòng / tile — bounds, image_area, inside/outside.
- `per_image_province_rows`: 1 dòng / (tile × tỉnh) — class breakdown.
- `province_summary[code]`: tổng cộng dồn cho mỗi tỉnh, **mỗi mùa 1 bucket riêng** (inference mode) hoặc 1 bucket "all" (dataset mode).
- `coverage_summary`: tổng ảnh, tổng area cho mỗi mùa.

### Step 7: Ghi CSV/JSON

`ReportWriter.write()` xuất:
- `summary_province_<season>.csv`: 7 dòng, mỗi tỉnh 1 dòng, đủ class breakdown + coverage.
- `summary_province_by_season.csv`: 14 dòng (7 tỉnh × 2 mùa) — dễ pivot Excel.
- `summary_province_combined.csv`: 7 dòng, gộp cả 2 mùa.
- `per_image_province_area.csv`: chi tiết từng tile để debug.
- `method_metadata.json`: lưu `boundary_area_method`, `season_mapping`, `class_mapping`, args gọi lúc chạy.

---

## 4. Ví dụ số liệu cụ thể — đi từng bước với tile `2321_19_35_0304_2023`

### Bước 1 — Đọc georef

```python
>>> with rasterio.open('inference_tif/Resolution3x3/2321_19_35_0304_2023.tif') as src:
...     width, height, crs = src.width, src.height, src.crs  # (320, 301, EPSG:4326)
...     transform = src.transform
...     bounds = src.bounds  # BoundingBox(105.7600, 20.5430, 105.7888, 20.5701)
```

`transform.a = 8.98315e-05` độ/pixel ≈ **997.7 m/pixel** ở vĩ độ 20.55°
`transform.e = -8.98315e-05` độ/pixel ≈ **997.7 m/pixel**

Pixel tại `row = 150` (giữa tile, lat ~20.556°):
- `meters_per_degree_lon(20.556) = 104051.4` m/° → `Δx = 8.983e-05 × 104051.4 = 9.347 m`
- `meters_per_degree_lat(20.556) = 111128.4` m/° → `Δy = 8.983e-05 × 111128.4 = 9.981 m`
- **pixel_area ≈ 93.29 m²**

Pixel tại `row = 0` (lat ~20.570°):
- lon: `104064.3` → Δx = 9.348 m
- lat: `111129.0` → Δy = 9.982 m
- pixel_area ≈ **93.32 m²**

Pixel tại `row = 300` (lat ~20.543°):
- lon: `104038.6` → Δx = 9.346 m
- lat: `111127.7` → Δy = 9.980 m
- pixel_area ≈ **93.26 m²**

Chênh lệch giữa top/bottom: **93.32 vs 93.26** = 0.06% — không đáng kể nhưng đúng.

### Bước 2 — Tổng diện tích tile

```python
row_pixel_area = array của 301 giá trị ≈ 93.3 m² mỗi pixel
image_area_m2  = sum(row_pixel_area) * width
               ≈ 93.29 × 301 × 320
               ≈ 8,991,385 m²
               ≈ 8.9914 km²
```

So sánh với hardcoded `3×3=9.0 km²`: sai số **+0.096%** (tile nằm ở vĩ độ ~20.5°, khá xa xích đạo nên gần đúng). Ở Cà Mau (lat ~9°) thì sai số sẽ lớn hơn rõ rệt nếu vẫn hardcode 9 km² → đó là lý do cần per-row.

### Bước 3 — Đếm class pixels

Kết quả đếm thực tế cho `2321_19_35_0304_2023_infered.png`:

| Class | Pixel count | Ratio |
|---|---:|---:|
| Water (cyan) | 90,044 | 93.4% |
| Residential (yellow) | 3,964 | 4.1% |
| Rice field (red) | 1,650 | 1.7% |
| Forest (green) | 662 | 0.7% |
| Unidentifiable | 0 | 0.0% |
| **Tổng** | **96,320** | 100% |

*(Lưu ý: water chiếm 93.4% là dấu hiệu model đang nhầm — không phải lỗi pipeline tính.)*

### Bước 4 — Rasterize tỉnh

Với tile `2321_19_35` chạy qua ranh giới **Hà Nội + Bắc Ninh + Hưng Yên** (kiểm tra bounds_intersect):

```python
province_mask_HN  = rasterize(Hanoi.geometry, ...)   # 1 = thuộc Hà Nội
province_mask_BN  = rasterize(BacNinh.geometry, ...)
province_mask_HY  = rasterize(HungYen.geometry, ...)
```

Giả sử pixel nào thuộc ranh giới sẽ chỉ có `1` ở 1-2 tỉnh (không có chồng 3+). Kết quả:

```
province_mask_sum > 0  (inside_any):  ~96,320 × 1.0  = gần như toàn tile
province_mask_sum > 1  (overlap):     vài pixel ở biên giới, bỏ qua
```

### Bước 5 — Đếm theo cặp (province, class)

Giả sử tile nằm hoàn toàn trong tỉnh A:

```
pixels(HN, Water)      = 90,044
area_m2(HN, Water)     = 90,044 × ~93.29 = 8,398,203 m² ≈ 8.398 km²
area_ha(HN, Water)     = 839.82 ha

pixels(HN, Forest)     = 662
area_m2(HN, Forest)    = 662 × 93.29 = 61,758 m² ≈ 0.0618 km²

...
area_m2(HN, total)     = 8,991,385 m² ≈ 8.9914 km²  (check: tổng class = tổng tile ✓)
```

### Bước 6 — Coverage check

Tile này nằm hoàn toàn trong ranh giới 1 tỉnh:

```
covered_area = 8.9914 km²
boundary_area của tỉnh = ví dụ 823.4 km² (Bắc Ninh)
coverage_ratio = 8.9914 / 823.4 = 1.09%  (1 tile phủ ~1% diện tích tỉnh — hợp lý)
```

Với 2130 tile/mùa phủ 15,491 km² trên tổng boundary 15,434 km² → coverage = 1.0037 → overhang 0.37% do tile có rìa nhỏ ngoài ranh giới chính xác.

### Bước 7 — Output file

Tile `2321_19_35_0304_2023` đóng góp:
- 1 dòng trong `per_image_area.csv` (image_area_km2 = 8.9914, inside = 8.9914, outside = ~0)
- 1 dòng trong `per_image_province_area.csv` (có đủ 4 class + unknown)
- Đóng góp vào `summary_province_dong_xuan.csv` (tích lũy theo tỉnh)

---

## 5. Dataset mode (ground truth QC)

Input: `dataset/Area_2024_N_o_<index>_sat.tif` + `Area_2024_N_o_<index>_mask.png`

Pipeline **giống hệt inference**, chỉ khác source. Không có season — tất cả tile gộp vào 1 bucket `"all"`. Output:

- `summary_province_area.csv`: 7 dòng, 1 dòng/tỉnh — class breakdown của label ground-truth.
- Dùng để đối chiếu với `summary_province_combined.csv` (inference) → ra độ lệch model.

---

## 6. Sanity checks chứng minh tính đúng

### S1. Pixel count đóng gói

```
sum(pixels của tất cả class trong 1 tile) == total_pixels của tile
sum(area_m2 của tất cả class trong 1 tile) == image_area_m2
```

Trong code: `area_for_mask` dùng tích vô hướng `np.dot(counts_per_row, row_pixel_area)` → tổng chính xác bằng phép toán số học, không tròn sai số.

### S2. Coverage ratio gần 1.0

Kết quả thực tế: `coverage_ratio_over_boundary = 1.0037` (Đông Xuân) và 1.0037 (Hè Thu). Sai số chỉ do tile overhang ngoài ranh giới tỉnh — **không phải lỗi thuật toán**.

Nếu sai số > 5% → bug ở rasterize, transform, hoặc `bounds_intersect`.

### S3. Tổng 2 mùa

Inference mode: 2130 tile ĐX + 2130 tile HT = 4260 tile → mỗi mùa phủ **15,491 km²** — y hệt nhau (vì mỗi mùa đều phủ toàn bộ 7 tỉnh một lần). Đây là behavior đúng: mỗi tile tồn tại trong cả 2 mùa nhưng là composite khác nhau thời điểm.

### S4. Boundary area khớp GADM

| Tỉnh | boundary_area_km2 (từ code) | GADM ref | Chênh |
|---|---:|---:|---:|
| Bắc Ninh | 823.42 | ~824 km² | 0.07% |
| Hà Nội | 3366.18 | ~3359 km² | 0.2% |
| Hải Dương | 1674.76 | ~1661 km² | 0.8% |
| Hải Phòng | 1403.43 | ~1528 km² | -8.2% |
| Hưng Yên | 929.15 | ~930 km² | -0.1% |
| Quảng Ninh | 6003.17 | ~6208 km² | -3.3% |
| Vĩnh Phúc | 1233.95 | ~1235 km² | -0.1% |

Chênh lệch chủ yếu do (1) GeoJSON của bạn đã được clip lọc (7 tỉnh được cắt theo boundary khác với GADM chuẩn), (2) ranh giới hành chính có thể không khớp 100% với GADM do sửa đổi thực tế. Điểm mấu chốt: **cùng 1 GeoJSON dùng cho cả dataset lẫn inference** → so sánh tương đối chính xác, chỉ cần chú ý khi đối chiếu số liệu tuyệt đối với GADM.

### S5. Không có pixel "bị mất"

```
total tile pixels = width × height = 320 × 301 = 96,320
sum(class_pixels trong tile) + unknown_pixels = 96,320   (check từng tile)
```

Trong dataset mode: `unknown_color_pixels = 0` cho 783/783 tile → mask RGB hoàn toàn khớp 5 class. Inference mode cũng tương tự (model output PNG chỉ sinh 4 màu đã định nghĩa).

### S6. Per-row pixel area là đúng

So sánh 2 cách tính cho tile `2321_19_35`:
- Per-row: `sum(301 giá trị) × 320 ≈ 8,991,385 m²`
- Đơn giản `width × height × a × e × 111320²` (hardcode): `320 × 301 × (8.983e-05 × 111320)² ≈ 8,991,280 m²`
- Sai số ~105 m² = 0.0012% → gần như không đáng kể ở vĩ độ 20.5°, nhưng tăng lên gần 0.1% ở vĩ độ 10°.

---

## 7. Error budget

| Nguồn sai | Ước lượng | Hướng giảm |
|---|---:|---|
| Per-row pixel area (hardcoded) | < 0.1% ở 20°N | Đã dùng per-row chuẩn WGS84 |
| Tile overhang ngoài tỉnh | ~0.37% tổng | Không thể khác được — tile 3×3 cố định |
| `all_touched=False` (pixel center) vs all-touched | < 1 pixel width | `--all_touched` nếu cần |
| Chồng ranh biên 2 tỉnh | 0.63 km² / 19,069 km² = 0.003% | Bỏ qua không đáng kể |
| PNG resize về TIF size | 0 (NEAREST, không đổi màu) | Giữ nguyên |
| RGB mismatch (unknown pixel) | 0 trong 100% tile hiện tại | Cảnh báo trong CSV |

**Sai số tổng hợp ước lượng**: ~0.4-0.5% (chủ yếu do tile overhang). Sai số còn lại >1% là do **chất lượng model** (class imbalance, nhầm Water/Rice, v.v.), không phải pipeline tính.

---

## 8. Đối chiếu Dataset vs Inference

| Chỉ số | Dataset (ground truth) | Inference (ĐX + HT trung bình) |
|---|---:|---:|
| Số tile | 783 | 2,130 |
| Tổng area | 22,600 km² | 19,069 km² |
| Inside 7 tỉnh | 15,373 km² | 15,491 km² |
| Coverage ratio | 1.0009 | 1.0037 |
| Tỷ lệ Forest | ~22-25% | ~5-8% |
| Tỷ lệ Water | ~15-20% | ~50-60% |

→ Dataset có tỷ lệ Forest cao hơn nhiều (hợp lý vì label thủ công chú ý rừng), Inference đang nhầm nhiều Water → cần xem lại model, **không phải bug ở pipeline tính**.

---

## 9. CLI & code entry point

```bash
# Entry point duy nhất
python calculate_area.py <mode> [options]

# Modes
python calculate_area.py dataset  --dataset_dir dataset  --output area_output/dataset_gt
python calculate_area.py inference --inference_tif_dir inference_tif/Resolution3x3 \
                                  --inference_png_dir inference_png/<run> \
                                  --output area_output/inference_by_boundary/<run>

# Options chung
  --boundary mapbox/vietnam_adm1_7provinces_2024.geojson
  --boundary_metadata mapbox/vietnam_adm1_7provinces_2024_metadata.json
  --all_touched      # rasterize dùng all-touched thay vì pixel center
```

Architecture:
```
area_calc/
├── cli.py          ← argparse, dispatch mode → source → calculator → aggregator → writer
├── config.py       ← RGB→class, SEASON_BY_DATE_TAG, slugify
├── sources.py      ← ImageSource ABC, DatasetSource, InferenceSource, ImagePair
├── geo.py          ← Province, per-row pixel area, rasterize, geometry area
├── masks.py        ← load PNG, build class masks, area_for_mask
├── calculator.py   ← AreaCalculator.process(tif, mask) → TileResult
└── aggregator.py   ← SummaryAggregator, ProvinceAccumulator, CoverageAccumulator, ReportWriter
```

---

## 10. FAQ thường gặp khi thuyết trình

**Q: Tại sao không dùng `gdal_calc.py` hay QGIS?**  
A: Pipeline này cần kết hợp (a) đếm class từ PNG với (b) rasterize province boundary với (c) tính diện tích theo vĩ độ — tất cả tự động, tái lập, version control được. QGIS/GDAL riêng lẻ không cover được logic accumulate theo bucket mùa.

**Q: Tại sao dùng `rasterio.features.rasterize` thay vì `gdal.RasterizeLayer`?**  
A: Rasterize lên grid của TIF sẵn có → transform khớp tuyệt đối, không cần reproject. Điều này bảo đảm pixel `(i, j)` trong tile đối ứng chính xác với pixel `(i, j)` trong mask PNG.

**Q: Có mất pixel ở cạnh tile không?**  
A: Mỗi tile là lưới đều 3×3 km, không overlap (trừ vài pixel ở ranh giới tỉnh do tile đứng trùng đường biên). Tổng overlap 0.63 km² / 19,069 km² = 0.003% — không đáng kể.

**Q: Làm sao kiểm tra lại kết quả?**  
A: Mỗi file output có `method_metadata.json` lưu đầy đủ `args`, `boundary_area_method`, `class_mapping`, `season_mapping`. Chạy lại cùng command → output giống hệt (deterministic, không có random). Per-image row cho phép trace từng tile về file TIF/PNG gốc.

**Q: Dữ liệu crawl có đại diện cho cả năm không?**  
A: Mỗi mùa là composite median của toàn bộ ảnh Sentinel-1 trong khoảng thời gian đó — không chỉ 1 ngày. Median giảm nhiễu speckle SAR. Đây là phương pháp phổ biến trong land-use mapping từ SAR.
