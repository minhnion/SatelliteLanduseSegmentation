# Phương pháp tính diện tích từ train dataset có label

Tài liệu này mô tả phương pháp tính diện tích các lớp land-use từ train dataset đã gắn nhãn. Nội dung tập trung vào input cần có, ý nghĩa từng loại dữ liệu, cách tính từ một ảnh cụ thể, cách tổng hợp theo tỉnh, các kiểm tra chứng minh phép tính đáng tin cậy và phần đối chiếu với diện tích lúa theo niên giám.

## 1. Mục tiêu

Train dataset gồm các cặp ảnh vệ tinh và mask label:

```text
dataset/Area_2024_N_o_0_sat.tif
dataset/Area_2024_N_o_0_mask.png
```

Mục tiêu là tính:

```text
Mỗi tỉnh có bao nhiêu ha Forest, Rice field, Water, Residential, Unidentifiable?
```

Phép tính không lấy toàn bộ diện tích hình chữ nhật của tile ảnh. Mỗi pixel được xác định theo ba câu hỏi:

```text
Pixel này nằm trong tỉnh nào?
Pixel này có label class gì?
Pixel này có diện tích bao nhiêu m2?
```

Sau đó diện tích được cộng theo cặp:

```text
province x class
```

Ví dụ kết quả cuối cùng cần có:

```text
Bắc Ninh - Rice field: 42,681 ha
Hà Nội - Rice field: 125,717 ha
Quảng Ninh - Forest: 413,698 ha
```

## 2. Input cần có

Phương pháp dùng ba loại input chính:

```text
1. GeoTIFF ảnh gốc: dataset/*_sat.tif
2. Mask label RGB: dataset/*_mask.png
3. Ranh giới tỉnh: mapbox/vietnam_adm1_7provinces_2024.geojson
```

### 2.1. GeoTIFF ảnh gốc

Ví dụ:

```text
dataset/Area_2024_N_o_0_sat.tif
```

Đây là ảnh vệ tinh gốc. File này quan trọng vì có thông tin địa lý. Nếu chỉ có PNG/JPG thông thường thì không biết mỗi pixel nằm ở đâu trên bản đồ. GeoTIFF thì có thể đọc được vị trí địa lý của ảnh.

Thông tin đọc được từ một file TIFF mẫu:

```text
driver: GTiff
band_count: 2
band descriptions: VV, VH
crs: EPSG:4326
width: 558
height: 558
bounds:
  left   = 105.249942
  bottom = 20.999916
  right  = 105.300068
  top    = 21.050042
```

Ý nghĩa từng trường:

`width`, `height`:

```text
Ảnh có 558 x 558 pixel.
```

`bounds`:

```text
Toàn bộ ảnh phủ một hình chữ nhật địa lý.
Kinh độ từ 105.249942 đến 105.300068.
Vĩ độ từ 20.999916 đến 21.050042.
```

`crs = EPSG:4326`:

```text
Tọa độ của ảnh là longitude/latitude, đơn vị là độ.
Vì vậy không thể lấy trực tiếp kích thước theo độ rồi coi là mét.
```

`transform`:

```text
Là công thức biến pixel row/col thành tọa độ longitude/latitude.
Nhờ transform, có thể biết pixel hàng i, cột j nằm ở đâu trên bản đồ.
```

`count`, `descriptions`:

```text
Ảnh có 2 band Sentinel-1: VV và VH.
Hai band này dùng cho train/model.
Đối với bài toán tính diện tích, thông tin quan trọng nhất từ TIFF là georeference.
```

### 2.2. Mask label RGB

Ví dụ:

```text
dataset/Area_2024_N_o_0_mask.png
```

Mask là ảnh label. Mỗi pixel trong mask có một màu RGB. Màu RGB đại diện cho class land-use.

Bảng màu:

| RGB | Class |
|---|---|
| `(0, 0, 0)` | Unidentifiable |
| `(0, 0, 255)` | Unidentifiable |
| `(255, 255, 255)` | Unidentifiable |
| `(0, 255, 0)` | Forest |
| `(255, 0, 0)` | Rice field |
| `(0, 255, 255)` | Water |
| `(255, 255, 0)` | Residential |

Ví dụ:

```text
mask[row=100, col=200] = (255, 0, 0)
=> pixel này được gắn nhãn Rice field.
```

Mask không tự cho biết diện tích. Mask chỉ cho biết class của từng pixel. Diện tích của pixel phải lấy từ GeoTIFF.

Nếu kích thước mask khác kích thước TIFF, mask được đưa về cùng size với TIFF bằng nearest-neighbor. Lý do dùng nearest-neighbor:

```text
Cần giữ nguyên màu label gốc.
Không tạo màu trung gian.
```

Nếu resize bằng nội suy mềm, màu `(255, 0, 0)` có thể thành `(254, 2, 0)`. Khi đó pixel không còn khớp với bảng class.

### 2.3. Ranh giới 7 tỉnh

File:

```text
mapbox/vietnam_adm1_7provinces_2024.geojson
```

File này chứa polygon ranh giới 7 tỉnh:

```text
Bắc Ninh
Hà Nội
Hải Dương
Hải Phòng
Hưng Yên
Quảng Ninh
Vĩnh Phúc
```

Ý nghĩa:

```text
Polygon tỉnh cho biết điểm nào thuộc tỉnh nào.
Pixel nằm ngoài polygon tỉnh sẽ không được cộng vào diện tích của tỉnh đó.
```

Ranh giới này được lấy từ GADM 4.1. Do nhóm cũ không còn boundary gốc, GADM 4.1 là nguồn thay thế hợp lý trong điều kiện dữ liệu hiện tại.

## 3. Ví dụ tính trên một ảnh

Giả sử cần tính diện tích `Rice field` của Bắc Ninh từ cặp file:

```text
Area_2024_N_o_0_sat.tif
Area_2024_N_o_0_mask.png
```

### Bước 1. Đọc vị trí địa lý của ảnh

Từ TIFF, lấy:

```text
width = 558
height = 558
bounds = bbox của ảnh
transform = pixel -> longitude/latitude
crs = EPSG:4326
```

Sau bước này, mỗi pixel của ảnh có thể được gắn với một tọa độ địa lý.

Ví dụ về mặt ý tưởng:

```text
pixel (0, 0) nằm gần góc trên trái của bbox
pixel (557, 557) nằm gần góc dưới phải của bbox
pixel (100, 200) có một tọa độ longitude/latitude cụ thể tính từ transform
```

### Bước 2. Đọc mask label

Mở file:

```text
Area_2024_N_o_0_mask.png
```

Mỗi pixel của mask tương ứng với pixel cùng hàng/cột của TIFF.

Ví dụ:

```text
mask[100, 200] = (255, 0, 0)
=> pixel (100, 200) là Rice field
```

### Bước 3. Đưa ranh giới tỉnh lên grid pixel của ảnh

Lấy polygon Bắc Ninh từ file GeoJSON. Polygon này đang ở tọa độ longitude/latitude.

Ảnh TIFF đang là grid pixel 558 x 558. Cần biến polygon Bắc Ninh thành một mask nhị phân cùng kích thước với ảnh:

```text
province_mask: 558 x 558
```

Mỗi pixel trong `province_mask` có giá trị:

```text
1 nếu tâm pixel nằm trong polygon Bắc Ninh
0 nếu tâm pixel nằm ngoài polygon Bắc Ninh
```

Ví dụ:

```text
province_mask[100, 200] = 1
=> pixel này thuộc Bắc Ninh

province_mask[300, 400] = 0
=> pixel này không thuộc Bắc Ninh
```

Nói cách khác:

```text
Đặt ranh giới Bắc Ninh lên ảnh vệ tinh.
Đánh dấu pixel nào nằm trong Bắc Ninh.
```

### Bước 4. Tạo mask cho từng class

Từ mask RGB, tạo riêng từng class mask.

Ví dụ với `Rice field`:

```text
rice_mask[row, col] = 1 nếu mask[row, col] = (255, 0, 0)
rice_mask[row, col] = 0 nếu không phải màu đỏ
```

Tương tự:

```text
forest_mask
water_mask
residential_mask
unidentifiable_mask
```

Lúc này có hai lớp thông tin quan trọng:

```text
province_mask: pixel nào thuộc tỉnh
rice_mask: pixel nào là Rice field
```

### Bước 5. Lấy giao giữa tỉnh và class

Muốn tính lúa trong Bắc Ninh:

```text
rice_in_bac_ninh = province_mask AND rice_mask
```

Ý nghĩa:

```text
Chỉ những pixel vừa nằm trong Bắc Ninh, vừa có màu đỏ Rice field mới được tính.
```

Ví dụ:

```text
province_mask[100, 200] = 1
rice_mask[100, 200] = 1
=> pixel này tính vào Rice field của Bắc Ninh
```

Ngược lại:

```text
province_mask[120, 300] = 0
rice_mask[120, 300] = 1
=> pixel này là Rice field trong mask, nhưng nằm ngoài Bắc Ninh
=> không tính cho Bắc Ninh
```

Đây là điểm quan trọng. Một tile ảnh có thể vượt ra ngoài ranh giới tỉnh, nhưng chỉ phần pixel nằm trong polygon tỉnh mới được cộng.

### Bước 6. Tính diện tích của từng pixel

TIFF dùng `EPSG:4326`, nghĩa là tọa độ theo độ kinh/vĩ. Vì vậy diện tích pixel không phải là một hằng số mét vuông lấy trực tiếp từ transform.

Với mỗi hàng pixel, lấy vĩ độ trung tâm của hàng đó:

```text
latitude_row
```

Sau đó tính:

```text
pixel_area_m2(row) =
pixel_width_degree * meters_per_degree_lon(latitude_row)
*
pixel_height_degree * meters_per_degree_lat(latitude_row)
```

Ý nghĩa:

```text
1 độ vĩ độ gần như có độ dài gần ổn định.
1 độ kinh độ thay đổi theo vĩ độ.
Vì vậy pixel ở các hàng khác nhau có thể có diện tích mét vuông hơi khác nhau.
```

Phương pháp này tốt hơn việc giả định:

```text
1 pixel = diện tích cố định
```

### Bước 7. Cộng diện tích các pixel hợp lệ

Sau khi có `rice_in_bac_ninh`, diện tích lúa của Bắc Ninh trong ảnh này là:

```text
rice_area_bac_ninh_m2 =
sum(pixel_area_m2(row) for pixel where rice_in_bac_ninh = 1)
```

Thực tế cộng theo từng hàng:

```text
for mỗi row:
    count = số pixel Rice field thuộc Bắc Ninh trên row đó
    area += count * pixel_area_m2(row)
```

Sau đó đổi đơn vị:

```text
area_km2 = area_m2 / 1,000,000
area_ha  = area_m2 / 10,000
```

### Bước 8. Lặp lại cho mỗi class

Với cùng polygon Bắc Ninh, tính lần lượt:

```text
forest_in_bac_ninh
rice_in_bac_ninh
water_in_bac_ninh
residential_in_bac_ninh
unidentifiable_in_bac_ninh
```

Mỗi class sẽ có diện tích riêng.

### Bước 9. Lặp lại cho mỗi tỉnh

Một ảnh có thể nằm trong một tỉnh hoặc cắt qua nhiều tỉnh. Phương pháp không gán cả ảnh cho một tỉnh duy nhất.

Với mỗi tỉnh:

```text
rasterize polygon tỉnh lên grid pixel của ảnh
lấy giao với từng class mask
tính diện tích
```

Nếu ảnh cắt qua ranh giới Bắc Ninh và Hà Nội:

```text
pixel nằm trong Bắc Ninh tính cho Bắc Ninh
pixel nằm trong Hà Nội tính cho Hà Nội
pixel nằm ngoài cả 7 tỉnh không tính vào tỉnh nào
```

### Bước 10. Lặp lại cho toàn bộ dataset

Phương pháp trên được thực hiện cho tất cả cặp ảnh:

```text
Area_2024_N_o_0
Area_2024_N_o_1
...
Area_2024_N_o_782
```

Sau đó cộng dồn theo tỉnh và class:

```text
total_rice_bac_ninh =
rice_bac_ninh_from_image_0
+ rice_bac_ninh_from_image_1
+ ...
+ rice_bac_ninh_from_image_782
```

Kết quả cuối cùng nằm trong:

```text
area_output/dataset_ground_truth_gadm_boundary/summary_province_area.csv
```

## 4. Coverage được tính như thế nào

Coverage trả lời câu hỏi:

```text
Dataset ảnh gốc phủ được bao nhiêu phần diện tích của tỉnh?
```

Coverage không dùng label class. Coverage chỉ dùng:

```text
GeoTIFF + polygon tỉnh
```

Với mỗi tỉnh:

```text
covered_area_km2 =
tổng diện tích pixel của các ảnh TIFF nằm trong polygon tỉnh
```

Sau đó:

```text
coverage_ratio_over_boundary =
covered_area_km2 / boundary_area_km2
```

Ví dụ Bắc Ninh:

```text
boundary_area_km2 = 823.42
covered_area_km2  = 824.15

coverage = 824.15 / 823.42 = 1.0009 = 100.09%
```

Ý nghĩa:

```text
Dataset gần như phủ đầy đủ Bắc Ninh.
```

Coverage hơi trên 100% là do rasterize biên pixel và sai số nhỏ trong cách tính diện tích polygon. Với mức quanh 100%, có thể hiểu là dataset phủ gần đầy đủ tỉnh.

## 5. Vì sao phương pháp này tốt hơn cách grid ratio cũ

Cách cũ dựa vào grid 3km:

```text
tile ảnh -> mapbox grid 3km -> ratio tỉnh -> chia diện tích
```

Rủi ro:

```text
Dataset train không được crawl bằng grid 3km.
Dataset train được crawl bằng các ô 0.05 độ.
Tile crawl là rectangle đầy đủ, có thể chứa phần ngoài tỉnh.
Ratio của grid 3km không phải ratio thật của tile dataset.
```

Phương pháp mới:

```text
pixel của GeoTIFF -> tọa độ thật -> polygon tỉnh -> class mask -> diện tích
```

Ưu điểm:

```text
Không cần đoán tile thuộc tỉnh nào.
Không cần dùng ratio trung gian.
Không tính phần ảnh nằm ngoài tỉnh.
Dùng chính georeference của ảnh cần tính.
Mỗi pixel được xác định bằng tọa độ thật của nó.
```

Một câu tóm tắt:

```text
Lấy từng pixel của mask label, dùng GeoTIFF để biết pixel nằm ở đâu,
dùng boundary để biết pixel thuộc tỉnh nào,
rồi cộng diện tích pixel theo từng class và từng tỉnh.
```

## 6. Output và ý nghĩa

Thư mục output:

```text
area_output/dataset_ground_truth_gadm_boundary/
```

File chính:

`summary_province_area.csv`:

```text
Mỗi dòng là một tỉnh.
Mỗi cột class là diện tích class đó trong tỉnh.
Đây là file chính để lấy số liệu báo cáo.
```

`per_image_province_area.csv`:

```text
Mỗi dòng là một cặp image x province.
Dùng để audit: ảnh nào đóng góp bao nhiêu diện tích vào tỉnh nào.
```

`per_image_area.csv`:

```text
Mỗi dòng là một ảnh.
Cho biết ảnh đó có bao nhiêu diện tích nằm trong 7 tỉnh và bao nhiêu nằm ngoài 7 tỉnh.
```

`coverage_summary.csv`:

```text
Tổng hợp coverage toàn dataset.
```

`missing_pairs.csv`:

```text
Kiểm tra có ảnh TIFF nào thiếu mask hoặc mask nào không ghép được không.
```

`method_metadata.json`:

```text
Lưu lại nguồn boundary, mapping màu class, tham số chạy và method.
Dùng để truy vết về sau.
```

## 7. Các dấu hiệu kiểm chứng phép tính

### 7.1. Cặp TIFF/mask đầy đủ

Kết quả hiện tại:

```text
TIFF: 783
Mask: 783
Missing pairs: 0
```

Ý nghĩa:

```text
Tất cả ảnh vệ tinh trong dataset đều có mask label tương ứng.
```

### 7.2. Màu label hợp lệ

Kết quả:

```text
unknown_color_inside_provinces_area_km2 = 0
```

Ý nghĩa:

```text
Trong phần nằm trong 7 tỉnh, các pixel mask đều thuộc bảng màu class đã định nghĩa.
Không có màu RGB lạ gây sai đếm class.
```

### 7.3. Coverage gần đầy đủ

Tổng 7 tỉnh:

```text
boundary_area = 15,434.06 km2
covered_area  = 15,374.06 km2
coverage      = 99.61%
```

Ý nghĩa:

```text
Dataset phủ gần đầy đủ vùng 7 tỉnh.
Nếu số diện tích class có vấn đề, nguyên nhân tiếp theo nên kiểm tra là chất lượng label hoặc định nghĩa class,
không phải do dataset thiếu phủ không gian lớn.
```

### 7.4. Phần ngoài tỉnh đã được loại

Tổng extent của ảnh:

```text
image_total_area = 22,600.25 km2
inside_7provinces = 15,373.43 km2
outside_7provinces = 7,226.82 km2
```

Ý nghĩa:

```text
Các tile ảnh có phần rectangle nằm ngoài ranh giới 7 tỉnh.
Phương pháp hiện tại đã loại phần này khỏi tổng diện tích theo tỉnh.
```

## 8. Kết quả coverage theo tỉnh

| Tỉnh | Boundary km2 | Covered km2 | Coverage |
|---|---:|---:|---:|
| Bắc Ninh | 823.42 | 824.15 | 100.09% |
| Hà Nội | 3366.18 | 3369.10 | 100.09% |
| Hải Dương | 1674.76 | 1674.69 | 100.00% |
| Hải Phòng | 1403.43 | 1388.37 | 98.93% |
| Hưng Yên | 929.15 | 927.83 | 99.86% |
| Quảng Ninh | 6003.17 | 5954.80 | 99.19% |
| Vĩnh Phúc | 1233.95 | 1235.13 | 100.10% |

Nhận xét:

```text
Coverage các tỉnh đều xấp xỉ 99-100%.
Hải Phòng và Quảng Ninh thấp hơn một chút, nhưng vẫn gần đầy đủ.
Một số tỉnh trên 100% nhẹ do sai số biên pixel và diện tích boundary.
```

## 9. Kết quả diện tích label theo tỉnh

Đơn vị: `km2`.

| Tỉnh | Forest | Rice field | Water | Residential | Unidentifiable |
|---|---:|---:|---:|---:|---:|
| Bắc Ninh | 8.72 | 426.81 | 110.15 | 256.21 | 22.25 |
| Hà Nội | 288.48 | 1257.17 | 328.66 | 1421.45 | 73.33 |
| Hải Dương | 197.55 | 792.88 | 107.08 | 495.28 | 81.89 |
| Hải Phòng | 182.05 | 450.48 | 149.27 | 563.48 | 43.09 |
| Hưng Yên | 2.26 | 483.94 | 104.66 | 311.47 | 25.50 |
| Quảng Ninh | 4136.98 | 421.31 | 568.01 | 493.74 | 334.75 |
| Vĩnh Phúc | 288.92 | 350.63 | 51.20 | 519.68 | 24.71 |

Tổng 7 tỉnh:

| Class | Diện tích ha | Tỷ lệ |
|---|---:|---:|
| Unidentifiable | 60,553 | 3.94% |
| Forest | 510,497 | 33.21% |
| Rice field | 418,322 | 27.21% |
| Water | 141,902 | 9.23% |
| Residential | 406,132 | 26.42% |

## 10. So sánh diện tích lúa với số liệu niên giám

Sau khi tính diện tích label `Rice field` trên train dataset, có thể đối chiếu với diện tích lúa theo niên giám năm 2024.

Đơn vị trong bảng: `ha`.

| Tỉnh | Rice field từ dataset | Diện tích lúa niên giám | Chênh lệch | Chênh lệch % |
|---|---:|---:|---:|---:|
| Bắc Ninh | 42,681 | 29,700 | +12,981 | +43.71% |
| Hà Nội | 125,717 | 82,500 | +43,217 | +52.38% |
| Hải Dương | 79,288 | 54,300 | +24,988 | +46.02% |
| Hải Phòng | 45,048 | 27,600 | +17,448 | +63.22% |
| Hưng Yên | 48,394 | 25,300 | +23,094 | +91.28% |
| Quảng Ninh | 42,131 | 14,900 | +27,231 | +182.76% |
| Vĩnh Phúc | 35,063 | 29,000 | +6,063 | +20.91% |
| **Tổng 7 tỉnh** | **418,322** | **263,300** | **+155,022** | **+58.88%** |

Cách đọc bảng:

```text
Rice field từ dataset = tổng diện tích pixel màu đỏ nằm trong boundary tỉnh.
Diện tích lúa niên giám = số liệu thống kê bên ngoài cho cùng tỉnh.
Chênh lệch = Rice field từ dataset - Diện tích lúa niên giám.
Chênh lệch % = Chênh lệch / Diện tích lúa niên giám.
```

Nhận xét từ kết quả so sánh:

```text
Tất cả 7 tỉnh đều có diện tích Rice field từ dataset cao hơn số liệu niên giám.
Tổng 7 tỉnh cao hơn 155,022 ha, tương đương +58.88%.
Quảng Ninh lệch lớn nhất theo tỷ lệ phần trăm: +182.76%.
Vĩnh Phúc lệch nhỏ nhất theo tỷ lệ phần trăm: +20.91%.
```

Kết quả này cho thấy sai lệch không phải là một lỗi cục bộ của riêng một tỉnh. Xu hướng lệch là đồng nhất: label `Rice field` trong dataset đang lớn hơn số liệu lúa niên giám ở tất cả các tỉnh.

### Kết luận về phương pháp tính

Phương pháp tính diện tích hiện tại là phương pháp tối ưu trong điều kiện dữ liệu hiện có, vì đã kiểm soát trực tiếp các yếu tố hình học quan trọng:

```text
1. Dùng georeference của chính từng file GeoTIFF gốc.
2. Dùng boundary 7 tỉnh để cắt pixel theo địa giới hành chính.
3. Chỉ tính pixel thật sự nằm trong polygon tỉnh.
4. Tính diện tích pixel theo vị trí địa lý của từng hàng ảnh.
5. Cộng dồn theo đúng class màu trong mask label.
6. Kiểm tra được coverage, missing pair và unknown color.
```

So với cách tính dựa trên grid ratio cũ, phương pháp này tốt hơn vì không cần suy luận ảnh thuộc tỉnh nào bằng tên file hay tỷ lệ ở lưới Mapbox. Mỗi pixel được quy về tỉnh dựa trên tọa độ thật của ảnh và polygon tỉnh.

Với coverage gần 99-100% cho từng tỉnh, missing pair bằng 0 và unknown color trong vùng tỉnh bằng 0, phần sai số do hình học, thiếu ảnh, thiếu mask hoặc màu label không phải là nguyên nhân chính của chênh lệch lớn với niên giám.

Kết luận quan trọng:

```text
Phép tính diện tích đã đáng tin cậy về mặt không gian.
Chênh lệch với niên giám chủ yếu cần được giải thích ở tầng dữ liệu label và định nghĩa thống kê.
```

Những khả năng cần xem xét khi giải thích chênh lệch:

```text
1. Label Rice field trong dataset có thể bao gồm nhiều loại đất nông nghiệp hơn riêng lúa.
2. Label có thể gán nhãn quá rộng, đặc biệt ở vùng có ảnh Sentinel-1 khó phân biệt.
3. Niên giám có thể tính diện tích gieo trồng, diện tích canh tác, hoặc diện tích lúa theo mùa vụ khác với định nghĩa mask.
4. Dataset có thể là label phủ bề mặt tại thời điểm/năm crawl, còn niên giám là thống kê hành chính tổng hợp.
5. Ranh giới GADM có thể khác nhẹ với ranh giới thống kê, nhưng mức lệch boundary này không đủ lớn để giải thích sai lệch +58.88%.
```

Vì vậy, khi báo cáo kết quả, nên trình bày rõ hai tầng kết luận:

```text
Tầng 1 - Phương pháp tính diện tích:
Đã được chuẩn hóa theo GeoTIFF, boundary tỉnh và diện tích pixel. Đây là cách tính phù hợp nhất với dữ liệu hiện có.

Tầng 2 - Chất lượng label:
Kết quả Rice field cao hơn niên giám ở cả 7 tỉnh, nên cần audit lại semantic label Rice field nếu mục tiêu là khớp với thống kê lúa chính thức.
```
