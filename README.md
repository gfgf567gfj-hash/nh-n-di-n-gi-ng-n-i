# Dự Án Điều Khiển Robot Bằng Giọng Nói

Dự án này sử dụng mô hình **SVM (Support Vector Machine)** kết hợp đặc trưng **MFCC (Mel-Frequency Cepstral Coefficients)** để nhận diện lệnh giọng nói tiếng Việt theo thời gian thực. Hệ thống thu âm trực tiếp từ microphone, trích xuất đặc trưng âm thanh và phân loại lệnh điều khiển robot/xe mô phỏng. Mô hình nhận diện chính xác 4 lệnh điều khiển:

- TIẾN (tien) — Di chuyển về phía trước
- TRÁI (trai) — Rẽ sang trái
- PHẢI (phai) — Rẽ sang phải
- DỪNG (dung) — Dừng lại

---

## 📂 Cấu trúc thư mục dự án

```
VoiceRobotProject/
│
├── my_dataset/               # Thư mục chứa tập dữ liệu giọng nói tự thu
│   ├── tien/                 # File .wav lệnh "TIẾN"
│   ├── trai/                 # File .wav lệnh "TRÁI"
│   ├── phai/                 # File .wav lệnh "PHẢI"
│   └── dung/                 # File .wav lệnh "DỪNG"
│
├── voice_model.pkl           # File mô hình SVM đã huấn luyện
├── scaler.pkl                # File chuẩn hóa dữ liệu StandardScaler
│
├── record_dataset.py         # Code thu âm và lưu dataset giọng nói
├── train.py                  # Code trích xuất đặc trưng MFCC và huấn luyện mô hình SVM
├── voice.py                  # Code nhận diện giọng nói thời gian thực (terminal)
├── robot_simulation.py       # Code mô phỏng xe robot điều khiển bằng giọng nói (Pygame)
│
├── requirements.txt          # Danh sách các thư viện cần cài đặt
└── README.md                 # Tài liệu hướng dẫn sử dụng dự án
```

---

## 🚀 Hướng dẫn vận hành chi tiết

**Bước 1** — Thu âm dataset giọng nói:
```
python record_dataset.py
```

**Bước 2** — Train lại mô hình:
```
python train.py
```

**Bước 3** — Test nhận diện giọng nói (terminal):
```
python voice.py
```

**Bước 4** — Chạy mô phỏng xe robot:
```
python robot_simulation.py
```

> 💡 Trong robot_simulation.py: nhấn **Enter** để nói lệnh, xe sẽ di chuyển theo lệnh nhận được.

---

## ⚙️ Yêu cầu hệ thống và Cài đặt môi trường

```
pip install -r requirements.txt
```

---

## 📦 Thư viện sử dụng

- **Librosa**: Trích xuất đặc trưng MFCC (20 hệ số), Delta và Delta-Delta từ tín hiệu âm thanh, chuẩn hóa biên độ và loại bỏ khoảng lặng (trim silence).
- **Scikit-learn**: Xây dựng mô hình SVM (kernel RBF, C=10), chuẩn hóa đặc trưng StandardScaler, đánh giá accuracy và classification report.
- **SoundDevice**: Thu âm trực tiếp từ microphone theo thời gian thực với sample rate tùy chỉnh.
- **NumPy**: Xử lý mảng tín hiệu số, tính RMS để phát hiện giọng nói, thêm nhiễu Gaussian cho data augmentation.
- **Pygame**: Xây dựng giao diện đồ họa 2D mô phỏng xe robot di chuyển trên màn hình theo lệnh giọng nói.
- **Joblib**: Lưu và nạp mô hình SVM cùng scaler dưới dạng file `.pkl`.
- **Scipy**: Ghi file âm thanh định dạng `.wav` trong quá trình thu dataset.

---

## 🗃️ Dataset

Dự án sử dụng bộ dữ liệu **tự thu âm** bằng `record_dataset.py`.

- Mỗi lệnh thu **30 file** âm thanh
- Thời lượng mỗi file: **1 giây**
- Sample rate: **16.000 Hz**
- Định dạng: **WAV (16-bit PCM)**
- 4 lớp lệnh: tien, trai, phai, dung
- Hỗ trợ xóa file vừa thu nếu thu sai (nhấn **d**)

---



## 📊 Kết quả huấn luyện

<img width="366" height="260" alt="image" src="https://github.com/user-attachments/assets/5274fbf1-b163-45fa-932c-ebfeef626e32" />
<img width="559" height="250" alt="image" src="https://github.com/user-attachments/assets/d9cfe837-0788-4cb8-90dc-6b2ae7379abd" />


---

## 🖥️ Ví dụ kết quả nhận diện

```
🔴 Đang ghi âm...
RMS = 0.0821
trai 94.3%

======================
⬅️ TRÁI
Độ tin cậy: 94.3%

Chi tiết:
tien :   2.10%
trai :  94.30%
phai :   2.80%
dung :   0.80%
======================
```

---

## 💾 File mô hình

- **voice_model.pkl** — Mô hình SVM đã huấn luyện
- **scaler.pkl** — Bộ chuẩn hóa StandardScaler khớp với tập train

> ⚠️ Hai file này phải được train cùng nhau và dùng cùng nhau. Không thay thế riêng lẻ.

HỌ TÊN:HUỲNH NGỌC TRƯỜNG

MSSV: 2421060243

LỚP: DCCDROBOT69
