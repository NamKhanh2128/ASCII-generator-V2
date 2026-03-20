# 🎨 ASCII Generator V2 (AI-Powered)

Dự án chuyển đổi hình ảnh thành nghệ thuật ASCII (ASCII Art) sử dụng mạng nơ-ron tích chập (CNN) được xây dựng bằng PyTorch. 

Khác với các phương pháp map độ sáng pixel truyền thống, phiên bản V2 này sử dụng Deep Learning để nhận diện các mảng ảnh (patches) và dự đoán ký tự ASCII phù hợp nhất, mang lại kết quả có độ chi tiết và sắc nét cao.

## 🚀 Tính năng nổi bật
* **Kiến trúc linh hoạt:** Tách biệt hoàn toàn phần xử lý dữ liệu, mô hình và vòng lặp huấn luyện.
* **Cấu hình tập trung:** Dễ dàng thay đổi các siêu tham số (Hyperparameters) qua file `configs/config.yaml`.
* **Sẵn sàng cho Cloud:** Hỗ trợ luồng huấn luyện mượt mà trên Kaggle (GPU) và trích xuất trọng số (weights) về chạy thực tế trên máy cá nhân.

## 📂 Cấu trúc dự án

```text
ASCII-generator-V2/
├── configs/            # File cấu hình YAML chứa siêu tham số
├── data/               # Dữ liệu ảnh đầu vào (raw) và đã xử lý (processed)
├── models/             # Thư mục chứa trọng số mô hình (.pth) sau khi train
├── notebooks/          # File Jupyter/Kaggle Notebook để thử nghiệm
├── src/                # Mã nguồn lõi (Dataset, Model, Utils)
├── tests/              # Unit test kiểm tra các hàm phụ trợ
├── main.py             # Điểm điều hướng chính của chương trình
└── requirements.txt    # Danh sách thư viện Python cần thiết