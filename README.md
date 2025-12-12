# 🎮 Computer Vision Control

Bộ sưu tập các ứng dụng Python sử dụng thị giác máy tính (Computer Vision) để điều khiển máy tính thông qua cử chỉ tay, chuyển động đầu và nháy mắt.

## 📋 Yêu cầu

```bash
pip install -r requirements.txt
```

## 📁 Danh sách ứng dụng

### 🔊 `volume.py` - Điều khiển âm lượng bằng tay
Điều chỉnh âm lượng hệ thống (macOS) bằng cử chỉ tay:
- Dùng **ngón cái** và **ngón trỏ** để điều chỉnh
- Khoảng cách giữa 2 ngón càng xa → âm lượng càng lớn
- Hiển thị thanh âm lượng và FPS trên màn hình

### 🖱️ `pointer.py` - Điều khiển chuột bằng tay + nháy mắt
Di chuyển con trỏ chuột và click bằng cử chỉ:
- **Ngón trỏ** → di chuyển con trỏ chuột
- **Nháy mắt trái** → click trái
- **Nháy mắt phải** → click phải

### 👁️ `eye-mouse.py` - Điều khiển chuột bằng đầu + mắt
Điều khiển chuột hoàn toàn không cần dùng tay:
- **Chuyển động đầu** → di chuyển con trỏ
- **Nháy mắt trái** → click trái
- **Nháy mắt phải** → click phải

### ✏️ `air-draw.py` - Vẽ trong không khí
Ứng dụng vẽ bằng cử chỉ tay:
- **Giơ ngón trỏ** → vẽ
- **Xòe bàn tay** (vẫy tay) → đổi màu
- **Giơ ngón giữa** → xóa canvas

### 👤 `detect-leaving.py` - Phát hiện sự hiện diện
Phát hiện và ghi log khi người dùng rời khỏi màn hình:
- Đăng ký khuôn mặt người dùng mới
- Phát hiện trạng thái ON_SCREEN / OFF_SCREEN
- Ghi log thời gian hiện diện vào file `presence_log.txt`

### 💀 `thanos-snap.py` - Thanos Snap (XÓA FILE)
⚠️ **CẢNH BÁO: Ứng dụng này sẽ XÓA file thật!**

Mô phỏng cú búng tay của Thanos:
1. Chọn thư mục mục tiêu
2. Thực hiện cử chỉ **chụm ngón cái và ngón trỏ** (snap)
3. Xóa ngẫu nhiên **50%** file trong thư mục đã chọn

### 🍺 `beer-drink-simulator.py` - Mô phỏng uống bia
Ứng dụng giải trí mô phỏng uống bia:
- Giơ tay để "cầm cốc bia"
- Đưa tay lại gần miệng để "uống"
- Animation cốc bia sẽ cạn dần
- Phím `r` để đổ đầy lại, `a` để chạy animation tự động

---

## 🚀 Cách chạy

```bash
python <tên_file>.py
```

Ví dụ:
```bash
python volume.py
python pointer.py
```

Nhấn `q` để thoát ứng dụng.

## 💻 Công nghệ sử dụng
- **OpenCV** - Xử lý hình ảnh & video
- **MediaPipe** - Nhận diện tay & khuôn mặt
- **PyAutoGUI** - Điều khiển chuột
- **NumPy** - Xử lý mảng số

## 📝 Ghi chú
- Các ứng dụng được thiết kế cho **macOS**
- Cần có webcam để sử dụng
- Đảm bảo đủ ánh sáng để nhận diện chính xác
