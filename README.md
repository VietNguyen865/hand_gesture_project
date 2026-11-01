# Hand Gesture Recognition for Human-Computer Interaction

Project môn Computer Vision: nhận diện cử chỉ tay từ webcam để phát triển hệ thống điều khiển PowerPoint không chạm.

## 1. Phạm vi giữa kỳ

Trong giữa kỳ, nhóm tập trung vào pipeline Computer Vision cổ điển:

```text
Video
  -> Tách frame
  -> Gán nhãn và chia dữ liệu
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
| `stop_palm` | Đưa bàn tay lên như ký hiệu dừng | Tạm dừng/che màn hình bằng phím `B` |
| `ok` | Tạo ký hiệu OK bằng ngón cái và ngón trỏ | Tiếp tục bằng phím `B` lần nữa |
| `neutral` | Không có tay hoặc tay ở trạng thái nghỉ | Không thực hiện lệnh |

`neutral` là lớp bắt buộc để hệ thống không gửi lệnh khi không có cử chỉ hợp lệ.

## 3. Cấu trúc thư mục

```text
hand_gesture_project/
|
|-- data/
|   |-- raw/
|   |   `-- videos/
|   |       |-- participant_01/
|   |       |   `-- session_01/
|   |       |       |-- p01_s01_thumbs_up.mp4
|   |       |       |-- p01_s01_thumbs_down.mp4
|   |       |       |-- p01_s01_stop_palm.mp4
|   |       |       |-- p01_s01_ok.mp4
|   |       |       `-- p01_s01_neutral.mp4
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
image_path,source_video,gesture,participant_id,session_id,lighting,background,hand_side,split
data/processed/frames/p01_s01_thumbs_up_000001.jpg,data/raw/videos/participant_01/session_01/p01_s01_thumbs_up.mp4,thumbs_up,participant_01,session_01,normal,plain,right,train
```

Trong đó:

- `image_path`: đường dẫn đến frame.
- `source_video`: video gốc tạo ra frame.
- `gesture`: một trong 5 label đã quy định.
- `participant_id`: mã người quay, không dùng tên thật.
- `session_id`: phiên quay dữ liệu.
- `lighting`: ví dụ `normal`, `low_light`, `backlight`.
- `background`: ví dụ `plain`, `cluttered`.
- `hand_side`: `left`, `right` hoặc `unknown`.
- `split`: `train`, `validation` hoặc `test`.

## 4. Quy tắc thu thập và đặt tên dữ liệu

### Video

Mỗi video chỉ nên chứa một cử chỉ. Thời lượng khuyến nghị là 2-5 giây; khoảng 3-4 giây là phù hợp cho cử chỉ tĩnh.

Trong video, người tham gia nên giữ cử chỉ ổn định ở phần giữa. Khi tách frame, ưu tiên lấy các frame ở đoạn ổn định và bỏ qua frame bắt đầu/kết thúc nếu bàn tay đang chuyển sang hoặc rời khỏi cử chỉ.

Quy tắc đặt tên:

```text
p<participant>_s<session>_<gesture>.mp4
```

Ví dụ:

```text
p01_s01_thumbs_up.mp4
p04_s02_stop_palm.mp4
```

Nếu cần ghi rõ điều kiện, có thể dùng:

```text
p04_s02_thumbs_down_lowlight_cluttered.mp4
```

### Frame

Quy tắc đặt tên:

```text
p<participant>_s<session>_<gesture>_<frame_number>.jpg
```

Ví dụ:

```text
p01_s01_thumbs_up_000001.jpg
p01_s01_thumbs_up_000002.jpg
```

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

### `02_segmentation_features.ipynb`

- So sánh HSV và Lab/K-means.
- Hiển thị binary mask.
- Thử morphology.
- Hiển thị contour, bounding box và crop.
- Kiểm tra các đặc trưng hình dạng.

### `03_classical_recognition_evaluation.ipynb`

- Huấn luyện kNN và SVM.
- So sánh kết quả classifier.
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
2. Chạy tách frame và tạo labels.csv
3. Chạy preprocessing
4. Chạy HSV/Lab segmentation
5. Chạy morphology và contour localization
6. Trích xuất Hu/shape features
7. Huấn luyện kNN/SVM
8. Chạy evaluation
9. Kiểm tra kết quả trong results/
```

Các phần CNN, temporal smoothing, realtime webcam và điều khiển PowerPoint sẽ được bổ sung ở cuối kỳ, không cần triển khai trong phạm vi giữa kỳ.

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
