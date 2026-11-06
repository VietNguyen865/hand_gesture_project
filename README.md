# Hand Gesture Recognition for Human-Computer Interaction

Project môn Computer Vision: nhận diện cử chỉ tay từ webcam để phát triển hệ thống điều khiển PowerPoint không chạm.

## 1. Phạm vi giữa kỳ

Trong giữa kỳ, nhóm tập trung vào pipeline Computer Vision cổ điển:

```text
Ảnh `hand_rgb` đã crop theo bbox
  -> Kiểm tra nhãn và split
  -> Tiền xử lý (màu, gamma, lọc ảnh, resize)
  -> HSV-V / Lab-K-means / Canny segmentation
  -> Morphology và contour
  -> Hu Moments / đặc trưng hình học hoặc HOG
  -> SVM
  -> Đánh giá macro-F1 và confusion matrix
```

Các phần CNN, xử lý theo thời gian, webcam real-time và điều khiển PowerPoint sẽ được phát triển ở cuối kỳ.

## 2. Các lớp cử chỉ

Giữa kỳ dùng các nhãn nguồn của MANUS-HaGRID và ánh xạ sang lệnh của project:

| Nhãn MANUS-HaGRID | Nhãn project | Ý nghĩa/lệnh dự kiến |
|---|---|---|
| `like` | `thumbs_up` | Sang slide tiếp theo |
| `dislike` | `thumbs_down` | Về slide trước |
| `stop` | `stop_palm` | Tạm dừng trình chiếu |
| `ok` | `ok` | Tiếp tục trình chiếu |
| `no_gesture` | `other` | Không phát lệnh |

`no_gesture` là trạng thái không có cử chỉ điều khiển hợp lệ; không nhất thiết là ảnh không có bàn tay. Giữ cả nhãn nguồn và nhãn project trong manifest để truy vết việc ánh xạ.

## 3. Cấu trúc thư mục và dữ liệu

```text
hand_gesture_project/
|
|-- data/
|   `-- manus_hagrid/            # Chỉ chứa dataset đã tải về
|       |-- data/                # WebDataset shards
|       |-- metadata/            # Manifest/schema đi kèm dataset
|       |-- docs/                # Tài liệu đi kèm dataset, nếu tải
|       `-- README.md            # Dataset card/ghi chú nguồn
|
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data_utils.py
|   |-- dataset.py               # Đọc shard/metadata và hand_rgb
|   |-- preprocessing.py         # Màu, gamma, filtering, resize
|   |-- augmentation.py
|   |-- segmentation.py          # HSV-V, Lab K-means, Canny, morphology
|   |-- features.py              # Hu/đặc trưng hình học và HOG
|   |-- recognition.py           # SVM/k-NN
|   |-- evaluation.py
|   `-- pipeline.py
|
|-- experiments/
|   |-- 01_data_preprocessing.ipynb
|   |-- 02_segmentation_features.ipynb
|   `-- 03_classical_recognition_evaluation.ipynb
|
|-- README.md
|-- requirements.txt
`-- .gitignore
```

### Dataset đầu vào

Giữa kỳ dùng [QFun/MANUS-HaGRID](https://huggingface.co/datasets/QFun/MANUS-HaGRID). Thư mục `data/manus_hagrid/` chỉ chứa các file tải từ dataset. Không đặt dữ liệu phát sinh từ thí nghiệm vào `data/`.

Notebook/loader đọc `hand_rgb.png`, là ảnh bàn tay đã crop sẵn theo bbox, cùng nhãn và metadata của sample. Nhóm không crop lại từ `full_rgb` và không đánh giá bước hand detection trong phạm vi giữa kỳ. Kiểm tra `available_files` và chỉ dùng các mẫu có `hand_rgb`. Test của `hagrid_v1` không có hand RGB dùng được; test `hagrid_v2` có hand crop ở 7,911/7,916 mẫu. Chọn một cấu hình nhất quán, chẳng hạn `hagrid_v2`, và loại các mẫu thiếu crop.

Metadata đi kèm dataset ghi nhãn nguồn, split, đường dẫn sample và các modality có sẵn. Dùng train/test split của dataset; chỉ tách validation từ train. Ánh xạ nhãn nguồn `like`, `dislike`, `stop`, `ok`, `no_gesture` sang nhãn lệnh của project trong `config.py` hoặc loader.

Không commit dataset shards lên GitHub. Giữ nguyên giấy phép và trích dẫn MANUS/HaGRID theo hướng dẫn của dataset.

### Kết quả trung gian

Ảnh crop đã có sẵn trong dataset. Mask, contour và feature được tạo trong notebook hoặc bộ nhớ khi chạy; không cần lưu thành các thư mục con dưới `data/`. Nếu cần xem lại kết quả, lưu hình/bảng trong output của notebook.

## 4. Nhãn và ảnh đầu vào

Các pipeline giữa kỳ nhận trực tiếp ảnh `hand_rgb` đã crop theo bbox và dùng nhãn/split đi kèm sample. Xem bảng ánh xạ nhãn nguồn sang lệnh của project ở mục 2.

## 5. Chia dữ liệu

- Giữ nguyên split train/test do MANUS-HaGRID cung cấp.
- Chỉ tách validation từ train; không chuyển mẫu test vào train/validation.
- Nếu cùng `source_image_id` hoặc cùng ảnh xuất hiện dưới nhiều dạng, giữ các bản liên quan trong cùng split.
- Chọn lớp và số lượng mẫu cân bằng trên train/validation; không dùng test để chọn ngưỡng, gamma hoặc tham số model.
- Mọi phương pháp phải dùng cùng danh sách `hand_rgb` và cùng split để so sánh.

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

Đọc shard MANUS-HaGRID, metadata, nhãn và ảnh `hand_rgb` thuộc `dataset.py`. Các tiện ích tổng quát khác chỉ dùng khi cần.

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

## 7. Ba notebook theo người phụ trách

Nhóm giữ ba notebook, mỗi notebook có một người phụ trách. Các thuật toán tái sử dụng đặt trong `src/`; notebook chứa cấu hình thí nghiệm, hình minh họa, bảng kết quả và phân tích của người phụ trách.

### `01_data_preprocessing.ipynb` — Người 1

- Kiểm tra nhãn và hiển thị các ảnh `hand_rgb` crop mẫu.
- Minh họa BGR/RGB, kênh HSV và kênh V.
- So sánh adaptive gamma bật/tắt trên nhánh HSV-V, giữ cùng ngưỡng và cùng split; kết luận chỉ giới hạn cho nhánh này.
- Thực hiện HSV-V → mask → morphology → contour → Hu/đặc trưng hình học → SVM.
- Phần Canny chung: khảo sát các ngưỡng thấp/cao.

### `02_segmentation_features.ipynb` — Người 2

- Chuyển Lab và thử K-means với K=2, 3, 4, 5; dùng quy tắc chọn cụm cố định, không chọn bằng mắt riêng cho từng ảnh.
- Thực hiện Lab → mask → morphology → contour → Hu/đặc trưng hình học → SVM.
- So sánh Median/Gaussian filtering và ghi ảnh hưởng lên mask.
- Phần Canny chung: khảo sát closing và lọc connected components.

### `03_classical_recognition_evaluation.ipynb` — Người 3

- Tạo baseline HOG trên crop grayscale → SVM.
- Tích hợp contour Canny → Hu/đặc trưng hình học → SVM thành hướng thứ tư.
- Phần Canny chung: tạo contour/mask, chạy phân loại và tổng hợp kết quả.
- So sánh bốn hướng trên cùng ảnh `hand_rgb`, split và metric; phân tích confusion matrix và lỗi.

Các hướng HSV-V, Lab-K-means và Canny dùng Hu Moments/đặc trưng hình học; hướng baseline dùng HOG. Cùng dùng SVM giúp so sánh nhất quán hơn. Đầu vào giữa kỳ đã là ảnh crop từ bbox MANUS; nhóm chưa đánh giá detector.

## 8. Quy tắc làm việc nhóm

- Không sửa trực tiếp code của người khác nếu chưa trao đổi.
- Mỗi thành viên có thể dùng branch riêng:
  - `feature/data-preprocessing`
  - `feature/segmentation`
  - `feature/recognition`
- Commit thường xuyên với nội dung rõ ràng, ví dụ:

```text
[DATA] load MANUS hand_rgb samples
[PREPROCESS] add adaptive gamma
[SEGMENT] add HSV mask
[FEATURE] add Hu moments
[MODEL] add SVM training
```

- Không commit các shard MANUS-HaGRID dung lượng lớn lên repository.
- Ghi nguồn, phiên bản, lớp được chọn và hướng dẫn tải dataset trong README.
- Kết quả thí nghiệm phải có thể tái tạo từ notebook, cấu hình và split đã ghi lại.

## 9. Quy trình chạy giữa kỳ

```text
1. Đọc metadata và kiểm tra các lớp like, dislike, stop, ok, no_gesture
2. Đọc `hand_rgb`; giữ split train/test chính thức
3. Lọc mẫu thiếu hand crop và tạo validation từ train
4. So sánh preprocessing cần thiết: hệ màu, gamma, filtering và resize
5. Chạy HSV-V segmentation, morphology, contour và Hu/đặc trưng hình học
6. Chạy Lab K-means segmentation, morphology, contour và Hu/đặc trưng hình học
7. Chạy HOG trên crop grayscale làm baseline
8. Chạy Canny, nối biên, contour, Hu/đặc trưng hình học
9. Huấn luyện cùng loại SVM và đánh giá macro-F1/confusion matrix
10. Phân tích các mẫu test đúng/sai và ghi nhận hạn chế khi chuyển sang webcam
```

Các hướng giữa kỳ là nhận dạng cổ điển, không dùng CNN làm mô hình chính. Webcam real-time, temporal smoothing và điều khiển PowerPoint thuộc phần phát triển sau.

## 10. Tăng cường dữ liệu (augmentation)

Augmentation là phần tùy chọn và chỉ áp dụng trên `hand_rgb` thuộc tập train sau khi đã cố định split. Giữ ảnh gốc và tạo biến thể trong bộ nhớ; không lưu biến thể vào `data/`.

- Không lật dọc hoặc xoay 180 độ mà giữ nguyên nhãn, vì có thể đổi `like` thành `dislike`.
- Kiểm tra ảnh sau biến đổi để chắc chắn ngón tay còn đủ và nhãn vẫn đúng.
- Lật ngang có thể đổi tay trái/phải; chỉ dùng nếu không ảnh hưởng mục tiêu phân loại.
- Ghi lại phép biến đổi, tham số và seed để tái tạo thí nghiệm.

```python
from src.augmentation import augment_image

# image là ảnh hand_rgb đã đọc thành mảng BGR uint8.
augmented_images = [augment_image(image, seed=42 + i) for i in range(3)]
```

Không augmentation validation hoặc test. Nếu dùng augmentation, báo riêng số lượng mẫu gốc và biến thể.

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

- [ ] `data/` chỉ chứa dataset MANUS-HaGRID đã tải về.
- [ ] Đầu vào là `hand_rgb` crop; đã lọc sample thiếu ảnh crop.
- [ ] Đã giữ split train/test chính thức và tạo validation từ train.
- [ ] Bốn hướng dùng cùng ảnh đầu vào, split và classifier SVM.
- [ ] Có minh họa kênh màu/chuyển đổi BGR-RGB, HSV/Lab/grayscale.
- [ ] Có thử adaptive gamma bật/tắt trên nhánh HSV-V và báo cáo kết quả.
- [ ] Có thử filtering/noise reduction và resize.
- [ ] Có morphology, segmentation, contour, Hu/đặc trưng hình học và HOG.
- [ ] Có macro-F1, confusion matrix và error analysis.
- [ ] Không báo IoU/Dice nếu chưa có mask ground truth.
- [ ] Không commit dataset shards lên GitHub.
- [ ] Có hướng dẫn chạy ba notebook.

