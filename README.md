# CS231 - NHẬN DIỆN PHƯƠNG TIỆN GIAO THÔNG TRÊN CAMERA ĐƯỜNG PHỐ

## 1. Thành viên

|   MSSV   |   Họ và tên    |
| :------: | :------------: |
| 22520414 | Hồ Trọng Hiển  |
| 22520121 |  Trần Gia Bảo  |
| 22520192 | Phạm Hồng Đăng |

## 2. Hướng dẫn chạy chương trình

-   Cài môi trường.

```bash
git clone https://github.com/hienhayho/yolov3-bkai.git

cd yolov3-bkai/

pip3 install poetry --user
poetry install
```

-   Tải checkpoint.

```bash
cd checkpoints/
bash ./download_checkpoint.sh
```

-   Chạy chương trình

```bash
cd ..
python app.py
```
