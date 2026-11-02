# Hand Gesture Recognition for Human-Computer Interaction

Project môn Computer Vision: nhận diện cử chỉ tay từ webcam để phát triển hệ thống điều khiển PowerPoint không chạm.

## 1. Phạm vi giữa kỳ

Trong giữa kỳ, nhóm tập trung vào pipeline Computer Vision cổ điển:

```text
Video
  -> Tách frame
  -> Gán nhãn và chia dữ liệu
  -> Augmentation (tùy chọn, chỉ tập train)
  -> Tiền xử lý
  -> Hand segmentation
  -> Morphology và connected components
  -> Contour, bounding box và crop bàn tay
  -> Trích xuất đặc trưng
  -> kNN/SVM
  -> Đánh giá
```

Các phần CNN, xử lý theo thời gian, webcam real-time và điều khiển PowerPoint sẽ được phát triển ở cuối kỳ.

## 2. Các lớp cử chỉ

Nhóm sử dụng 5 lớp:

| Label | Ý nghĩa | Lệnh dự kiến |
|---|---|---|
| `thumbs_up` | Giơ ngón cái lên | Sang slide tiếp theo |
| `thumbs_down` | Giơ ngón cái xuống | Về slide trước |
| `stop_palm` | Đưa bàn tay lên như ký hiệu dừng | Tạm dừng trình chiếu |
| `ok` | Tạo ký hiệu OK bằng ngón cái và ngón trỏ | Tiếp tục trình chiếu |
| `other` | Mọi trạng thái không thuộc 4 cử chỉ điều khiển trên | Không thực hiện lệnh |

Nhóm dùng lớp `other` để biểu diễn các trường hợp không thực hiện lệnh: không có tay, tay ngoài khung hình, tay nghỉ, cử chỉ không hợp lệ, tay bị che hoặc quá nhỏ để nhận dạng. Đây là một lớp duy nhất, không chia thành các lớp riêng cho từng trạng thái.

## 3. Cấu trúc thư mục

```text
hand_gesture_project/
|
|-- data/
|   |-- raw/
|   |   `-- videos/
|   |       |-- participant_01/
|   |       |   `-- session_01/
|   |       |       |-- p01_s01_thumbs_up_front.mp4
|   |       |       |-- p01_s01_thumbs_down_left.mp4
|   |       |       |-- p01_s01_stop_palm_right.mp4
|   |       |       |-- p01_s01_ok_front.mp4
|   |       |       `-- p01_s01_other_no_hand.mp4
|   |       |-- participant_02/
|   |       |-- participant_03/
|   |       `-- participant_04/
|   |
|   |-- processed/
|   |   `-- frames/
|   |
|   `-- labels.csv
|
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data_utils.py
|   |-- preprocessing.py
|   |-- augmentation.py
|   |-- segmentation.py
|   |-- features.py
|   |-- recognition.py
|   |-- evaluation.py
|   `-- pipeline.py
|
|-- experiments/
|   |-- 01_data_preprocessing.ipynb
|   |-- 02_segmentation_features.ipynb
|   `-- 03_classical_recognition_evaluation.ipynb
|
|-- results/
|   |-- figures/
|   |-- tables/
|   `-- models/
|
|-- README.md
|-- requirements.txt
`-- .gitignore
```

### `data/raw/`

Chứa dữ liệu gốc, chưa chỉnh sửa. Video trong thư mục này không được ghi đè.

### `data/processed/frames/`

Chứa các frame đã được tách từ video và làm sạch. Nếu cần lưu kết quả trung gian, có thể tạo thêm các thư mục sau trong `processed/`:

```text
processed/
|-- frames/
|-- augmented/  # Tùy chọn, chỉ lưu biến thể của tập train
|-- masks/
|   |-- hsv/
|   `-- lab_kmeans/
|-- crops/
`-- features/
```

Không bắt buộc phải lưu mọi mask, crop hoặc feature. Những dữ liệu này có thể được tạo lại bằng code khi cần.

### `data/labels.csv`

File quản lý nhãn và thông tin của từng frame. Dùng đường dẫn tương đối, không dùng đường dẫn tuyệt đối như `D:\\...`.

Schema đề xuất:

```csv
image_path,source_video,gesture,participant_id,session_id,view_angle,split
data/processed/frames/p01_s01_thumbs_up_front/participant_01_session_01_thumbs_up_front_000000.jpg,data/raw/videos/participant_01/session_01/p01_s01_thumbs_up_front.mp4,thumbs_up,participant_01,session_01,front,train
data/processed/frames/p01_s01_other_no_hand/participant_01_session_01_other_000000.jpg,data/raw/videos/participant_01/session_01/p01_s01_other_no_hand.mp4,other,participant_01,session_01,,train
```

Trong đó:

- `image_path`: đường dẫn đến frame.
- `source_video`: video gốc tạo ra frame.
- `gesture`: một trong 5 label đã quy định.
- `participant_id`: mã người quay, không dùng tên thật.
- `session_id`: phiên quay dữ liệu.
- `view_angle`: góc nhìn của bàn tay, gồm `front`, `left` hoặc `right`; được phép để trống cho lớp `other`. Trong Python dùng `None`, trong CSV để ô trống, không ghi chữ `empty`.
- `split`: `train`, `validation` hoặc `test`.

## 4. Quy tắc thu thập và đặt tên dữ liệu

### Video

Mỗi video chỉ nên chứa một cử chỉ. Thời lượng khuyến nghị là 2-5 giây; khoảng 3-4 giây là phù hợp cho cử chỉ tĩnh.

Trong video, người tham gia nên giữ cử chỉ ổn định ở phần giữa. Khi tách frame, ưu tiên lấy các frame ở đoạn ổn định và bỏ qua frame bắt đầu/kết thúc nếu bàn tay đang chuyển sang hoặc rời khỏi cử chỉ.

Quy tắc đặt tên cho các lớp cử chỉ điều khiển:

```text
p<participant>_s<session>_<gesture>_<view_angle>.mp4
```

Ví dụ:

```text
p01_s01_thumbs_up_front.mp4
p01_s01_thumbs_up_left.mp4
p01_s01_thumbs_up_right.mp4
p04_s02_stop_palm_left.mp4
p04_s02_ok_front.mp4
```

Ba góc quay được sử dụng là:

- `front`: bàn tay hướng tương đối trực diện camera.
- `left`: xoay khoảng 30-45 độ sang trái.
- `right`: xoay khoảng 30-45 độ sang phải.

Đối với lớp `other`, có thể thêm mô tả trạng thái vào tên video:

```text
p01_s01_other_no_hand.mp4
p01_s01_other_hand_outside.mp4
p01_s01_other_relaxed.mp4
```

Các hậu tố `no_hand`, `hand_outside`, `relaxed` chỉ mô tả video, không phải lớp hoặc góc nhìn mới. Cả ba video đều có `gesture=other` và để trống `view_angle` trong `labels.csv`.

Khi quay nhiều lần với cùng người, phiên quay, cử chỉ và góc nhìn (hoặc cùng mô tả trạng thái của lớp `other`), thêm hậu tố `_take01`, `_take02`, ... trước phần mở rộng để phân biệt các lần quay:

```text
p01_s01_thumbs_up_front_take01.mp4
p01_s01_thumbs_up_front_take02.mp4
p01_s01_other_no_hand_take01.mp4
p01_s01_other_no_hand_take02.mp4
```

Có thể dùng `_take01` ngay từ lần quay đầu tiên. Tên video không tính phần mở rộng phải duy nhất trong bộ dữ liệu, vì mỗi video được tách frame vào thư mục mang tên đó. Hậu tố `take` chỉ là số lần quay, không tạo thêm lớp, góc nhìn hoặc cột trong `labels.csv`.

### Frame

Hàm `extract_frames()` lưu mỗi video vào một thư mục riêng trong `data/processed/frames/`, đặt theo tên video không có phần mở rộng. Tên frame dùng trực tiếp `participant_id` và `session_id` được truyền vào hàm:

```text
<participant_id>_<session_id>_<gesture>_<view_angle>_<frame_number>.jpg
```

Ví dụ với `participant_id="participant_01"`, `session_id="session_01"`:

```text
p01_s01_thumbs_up_front/participant_01_session_01_thumbs_up_front_000000.jpg
p01_s01_thumbs_up_front/participant_01_session_01_thumbs_up_front_000005.jpg
```

Với lớp `other` và `view_angle=None`, tên frame bỏ phần góc nhìn. Trạng thái mô tả vẫn nằm trong tên thư mục video:

```text
p01_s01_other_no_hand/participant_01_session_01_other_000000.jpg
p01_s01_other_relaxed/participant_01_session_01_other_000000.jpg
```

Nếu tên video có hậu tố `_take01` hoặc `_take02`, thư mục frame cũng giữ hậu tố đó. Tên từng frame vẫn theo quy tắc trên, ví dụ:

```text
p01_s01_thumbs_up_front_take01/participant_01_session_01_thumbs_up_front_000000.jpg
p01_s01_thumbs_up_front_take02/participant_01_session_01_thumbs_up_front_000000.jpg
```

Số frame bắt đầu từ 0; mặc định lấy một frame sau mỗi 5 frame theo cấu hình.

Không lấy tất cả frame liên tiếp nếu chúng gần như giống nhau. Nên lấy frame đại diện và loại frame trùng hoặc quá nhòe.

### Người tham gia

- `participant_01` đến `participant_03`: thành viên nhóm.
- `participant_04` trở đi: người được mời quay thêm.
- Không dùng tên thật trong tên file hoặc `labels.csv`.
- Nên có sự đồng ý của người tham gia trước khi sử dụng dữ liệu.

## 5. Chia dữ liệu

Không chia ngẫu nhiên từng frame của cùng một video vào cả train và test. Các frame gần nhau có thể làm kết quả bị cao giả tạo.

Ưu tiên chia theo người:

```text
Train:      participant_01, participant_02, participant_03
Validation: participant_04
Test:       participant_05
```

Nếu chưa có đủ người, chia theo session và ghi rõ hạn chế này trong báo cáo.

Chia dữ liệu trước khi augmentation. Ảnh gốc và mọi biến thể của ảnh đó phải ở cùng tập train; không đưa biến thể sang validation hoặc test.

## 6. Chức năng của từng file trong `src/`

### `__init__.py`

Đánh dấu `src` là một Python package. File này chỉ chứa thông tin package, không chứa thuật toán xử lý ảnh. Nhờ đó các module có thể được import rõ ràng, ví dụ:

```python
from src.preprocessing import preprocess_image
from src.segmentation import hsv_segmentation
```

### `config.py`

Chứa các cấu hình dùng chung:

- Danh sách label.
- Kích thước ảnh.
- Ngưỡng segmentation.
- Tham số gamma.
- Đường dẫn dữ liệu.
- Tên phương pháp đang sử dụng.

Không đặt thuật toán chính trong file này.

### `data_utils.py`

Phụ trách dữ liệu:

- Đọc video và ảnh.
- Tách frame.
- Loại frame trùng.
- Kiểm tra ảnh lỗi.
- Tạo và đọc `labels.csv`.
- Tạo train/validation/test split.

### `preprocessing.py`

Phụ trách tiền xử lý:

- Resize.
- Chuyển đổi BGR/RGB, HSV và Lab.
- Median/Gaussian filtering.
- Gamma correction hoặc adaptive gamma.
- Contrast enhancement.
- Biến đổi hình học như crop, rotation hoặc scaling.

### `augmentation.py`

Tạo thêm biến thể từ ảnh BGR uint8 thuộc tập train:

- `random_rotate()`: xoay ngẫu nhiên, mặc định từ -15 đến 15 độ.
- `random_translate()`: dịch chuyển, mặc định tối đa 20 pixel theo mỗi trục.
- `random_scale()`: phóng to/thu nhỏ quanh tâm, mặc định từ 0.9 đến 1.1.
- `random_shear()`: shear ngang, mặc định từ -0.15 đến 0.15.
- `random_flip()`: chỉ lật ngang, xác suất mặc định 0.5.
- `augment_image()`: chọn một phép hình học trong bốn phép đầu, sau đó có thể lật ngang. Tham số `seed` giúp tái tạo kết quả.

Các hàm giữ nguyên kích thước, thứ tự màu BGR và kiểu uint8 của ảnh. Phần ngoài ảnh được đệm đen khi biến đổi hình học. Xem quy trình sử dụng ở mục 10.

### `segmentation.py`

Phụ trách tách và định vị bàn tay:

- HSV skin-color segmentation.
- Lab + K-means segmentation.
- Binary mask.
- Morphological opening/closing.
- Hole filling.
- Connected-component labeling.
- Contour detection.
- Bounding box và hand crop.

### `features.py`

Phụ trách trích xuất đặc trưng:

- Area.
- Perimeter.
- Centroid.
- Aspect ratio.
- Extent.
- Solidity.
- Convex hull.
- Hu moments.
- HOG nếu nhóm sử dụng.

### `recognition.py`

Phụ trách nhận dạng cổ điển:

- Feature scaling.
- kNN.
- SVM.
- Train, predict và lưu model.

### `evaluation.py`

Phụ trách đánh giá:

- IoU của mask.
- Precision, recall và F1.
- Accuracy và macro-F1.
- Confusion matrix.
- Runtime/FPS.
- Error analysis.

### `pipeline.py`

Ghép các module thành pipeline hoàn chỉnh:

```text
image
  -> preprocessing
  -> segmentation
  -> localization
  -> feature extraction
  -> recognition
  -> evaluation
```

Không nên viết lại toàn bộ thuật toán trong `pipeline.py`; file này chỉ điều phối các hàm từ module khác.

## 7. Chức năng của các notebook

### `01_data_preprocessing.ipynb`

- Kiểm tra frame và nhãn.
- Hiển thị RGB/BGR, HSV và Lab.
- So sánh filtering, gamma correction và các phép biến đổi ảnh.
- Hiển thị ảnh gốc và ảnh augmentation của tập train để kiểm tra cử chỉ còn rõ.

### `02_segmentation_features.ipynb`

- So sánh HSV và Lab/K-means.
- Hiển thị binary mask.
- Thử morphology.
- Hiển thị contour, bounding box và crop.
- Kiểm tra các đặc trưng hình dạng.

### `03_classical_recognition_evaluation.ipynb`

- Huấn luyện kNN và SVM.
- So sánh kết quả classifier.
- Nếu dùng augmentation, so sánh huấn luyện có/không augmentation trên cùng tập validation gốc.
- Tạo confusion matrix.
- Tính các chỉ số đánh giá.
- Phân tích trường hợp nhận diện sai.

Notebook chỉ dùng để thử nghiệm, trực quan hóa và tạo kết quả. Các hàm có thể tái sử dụng phải đặt trong `src/`.

## 8. Quy tắc làm việc nhóm

- Không sửa trực tiếp code của người khác nếu chưa trao đổi.
- Mỗi thành viên có thể dùng branch riêng:
  - `feature/data-preprocessing`
  - `feature/segmentation`
  - `feature/recognition`
- Commit thường xuyên với nội dung rõ ràng, ví dụ:

```text
[DATA] add frame extraction
[PREPROCESS] add adaptive gamma
[SEGMENT] add HSV mask
[FEATURE] add Hu moments
[MODEL] add SVM training
```

- Không commit video dung lượng lớn hoặc dữ liệu chứa thông tin cá nhân lên repository công khai.
- Nếu dữ liệu quá lớn, ghi hướng dẫn tải dữ liệu trong README hoặc cung cấp script tải dữ liệu.
- Mọi kết quả trong `results/` phải có thể tái tạo từ code và dữ liệu.

## 9. Quy trình chạy giữa kỳ

```text
1. Đặt video gốc vào data/raw/videos/
2. Tách frame, kiểm tra và gán nhãn trong labels.csv
3. Chia train/validation/test theo người hoặc session
4. Nếu dùng augmentation, tạo biến thể chỉ từ tập train
5. Chạy preprocessing cho ảnh gốc và ảnh augmentation
6. Chạy HSV/Lab segmentation
7. Chạy morphology và contour localization
8. Trích xuất Hu/shape features
9. Huấn luyện kNN/SVM bằng tập train
10. Chọn tham số bằng validation, đánh giá cuối trên test
11. Kiểm tra kết quả trong results/
```

Các phần CNN, temporal smoothing, realtime webcam và điều khiển PowerPoint sẽ được bổ sung ở cuối kỳ, không cần triển khai trong phạm vi giữa kỳ.

## 10. Tăng cường dữ liệu (augmentation)

Augmentation là phần mở rộng tùy chọn của nhóm. Giữ ảnh gốc trong tập train và thêm biến thể có cùng nhãn nếu cử chỉ vẫn rõ sau biến đổi.

- Chỉ augmentation sau khi đã chia dữ liệu và chỉ áp dụng cho tập train. Validation và test dùng ảnh gốc với cùng các bước tiền xử lý xác định như khi chạy hệ thống.
- Không lật dọc hoặc xoay 180 độ rồi giữ nguyên nhãn, vì có thể đổi `thumbs_up` thành `thumbs_down`.
- Kiểm tra ảnh sinh ra: biến đổi trên khung cố định có thể cắt mất ngón tay hoặc làm cử chỉ không còn rõ. Loại bỏ các biến thể đó thay vì tự động giữ nhãn.
- Các phép xoay trong mặt phẳng, dịch chuyển, scale, shear và lật ngang không tạo thêm giá trị `view_angle` mới. Biến thể hợp lệ kế thừa góc nhìn của ảnh nguồn; lớp `other` có thể để trống góc nhìn.

Ví dụ tạo ba biến thể từ một frame của tập train trong notebook:

```python
from src.augmentation import augment_image

# image là ảnh BGR uint8 đã đọc từ một frame thuộc tập train.
augmented_images = [
    augment_image(image, seed=42 + i)
    for i in range(3)
]
```

Cùng ảnh và seed sẽ cho cùng kết quả. Khi xử lý nhiều ảnh, dùng seed khác nhau cho từng ảnh/biến thể và lưu quy tắc seed để tái tạo thí nghiệm.

Hàm `augment_image()` chỉ trả về ảnh trong bộ nhớ, chưa tự lưu file hoặc cập nhật CSV. Có thể dùng trực tiếp ảnh đó cho các bước tiếp theo. Nếu muốn lưu ra đĩa, dùng thư mục riêng như:

```text
data/processed/augmented/p01_s01_thumbs_up_front_take01/
    participant_01_session_01_thumbs_up_front_000000_aug01.jpg
    participant_01_session_01_thumbs_up_front_000000_aug02.jpg
```

Tên biến thể giữ tên frame nguồn và thêm `_aug01`, `_aug02`, ...; thư mục cha giữ tên video để tránh trùng giữa các lần quay. Khi lưu biến thể vào `labels.csv`, vẫn dùng đúng 7 cột: cập nhật `image_path`, giữ `source_video`, `gesture`, `participant_id`, `session_id`, `view_angle` của ảnh nguồn và đặt `split=train`. Không cần thêm lớp hoặc cột mới.

Trong báo cáo, ghi số ảnh train gốc và số ảnh augmentation riêng, mô tả phép biến đổi, tham số và so sánh kết quả có/không augmentation. Số ảnh validation/test không tăng do augmentation.

## 11. Cách chạy package

Các lệnh nên được chạy từ thư mục gốc `hand_gesture_project/`.

Ví dụ chạy pipeline:

```bash
python -m src.pipeline
```

Nếu dùng import tương đối bên trong package, ví dụ:

```python
from .preprocessing import preprocess_image
```

thì không nên chạy trực tiếp bằng `python src/pipeline.py`; hãy chạy bằng dạng module `python -m src.pipeline`.

## 12. Checklist trước khi nộp code giữa kỳ

- [ ] Có video hoặc hướng dẫn tải dữ liệu.
- [ ] Có frame và `labels.csv`.
- [ ] Có ít nhất 5 lớp gesture.
- [ ] Có train/validation/test split.
- [ ] Nếu dùng augmentation: chỉ áp dụng trên train, kiểm tra nhãn và ghi lại tham số/seed.
- [ ] Có RGB/BGR và HSV/Lab conversion.
- [ ] Có filtering hoặc noise reduction.
- [ ] Có gamma/contrast enhancement hoặc phép tiền xử lý tương đương.
- [ ] Có morphology.
- [ ] Có segmentation.
- [ ] Có contour và bounding box.
- [ ] Có feature extraction.
- [ ] Có kNN hoặc SVM.
- [ ] Có evaluation và error analysis.
- [ ] Có hướng dẫn chạy trong README.
