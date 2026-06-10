# tests/test_train.py
import os
import torch
import tempfile
import shutil
from PIL import Image
from src.train import train_model

def test_training_pipeline():
    """Kiểm tra toàn bộ luồng huấn luyện bao gồm cả Validation và vẽ biểu đồ Loss."""
    # 1. Tạo thư mục tạm để chứa dữ liệu giả lập
    with tempfile.TemporaryDirectory() as temp_data_dir:
        processed_dir = os.path.join(temp_data_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        # Tạo 3 nhãn (0, 1, 2) mỗi nhãn có 2 ảnh patch kích thước 32x32
        image_size = 32
        for label in ["0", "1", "2"]:
            label_dir = os.path.join(processed_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            for i in range(2):
                img = Image.new('L', (image_size, image_size), color=(int(label) * 80 + i * 20))
                img.save(os.path.join(label_dir, f"img_{i}.png"))
                
        # 2. Thiết lập config giả lập
        with tempfile.TemporaryDirectory() as temp_save_dir:
            config = {
                'project_name': 'Test Train Project',
                'data': {
                    'processed_dir': processed_dir,
                    'image_size': image_size,
                    'val_split': 0.3 # 30% làm validation
                },
                'model': {
                    'batch_size': 2,
                    'learning_rate': 0.01,
                    'epochs': 2,
                    'num_classes': 3,
                    'save_dir': temp_save_dir
                }
            }
            
            # 3. Chạy luồng huấn luyện thực tế với cấu hình giả lập
            print("\n>>> Đang chạy thử nghiệm huấn luyện với dữ liệu giả lập...")
            train_model(config)
            
            # 4. Kiểm tra các file kết quả đầu ra
            best_model_path = os.path.join(temp_save_dir, "ascii_cnn_model.pth")
            last_model_path = os.path.join(temp_save_dir, "ascii_cnn_last_model.pth")
            plot_path = os.path.join(temp_save_dir, "loss_plot.png")
            
            assert os.path.exists(best_model_path), "LỖI: Không tìm thấy file checkpoint model tốt nhất!"
            assert os.path.exists(last_model_path), "LỖI: Không tìm thấy file checkpoint model cuối cùng!"
            assert os.path.exists(plot_path), "LỖI: Không tìm thấy file ảnh biểu đồ Loss!"
            
            print("✔ [PASS] test_training_pipeline: Vòng lặp huấn luyện, validation, checkpoint và vẽ biểu đồ hoạt động hoàn hảo.")

if __name__ == "__main__":
    print("="*40)
    print(">>> BẮT ĐẦU CHẠY TRAINING UNIT TEST...")
    print("="*40)
    test_training_pipeline()
    print("="*40)
    print(">>> TRAINING UNIT TEST HOÀN TẤT VÀ THÀNH CÔNG!")
