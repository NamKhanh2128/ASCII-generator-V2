import argparse
import yaml
import sys
from src.train import train_model
from src.inference import run_inference

def load_config(config_path):
    """Đọc các thông số cấu hình từ file YAML."""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"[LỖI] Không tìm thấy file cấu hình tại: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as exc:
        print(f"[LỖI] Cú pháp file YAML không hợp lệ. Vui lòng kiểm tra lại: {exc}")
        sys.exit(1)

def main():
    # 1. Khởi tạo bộ phân tích tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Chương trình chính của ASCII-Generator-V2")
    
    # 2. Khai báo các tham số bắt buộc và tự chọn
    parser.add_argument(
        '--mode', 
        type=str, 
        required=True, 
        choices=['train', 'inference'], 
        help="Chọn luồng chạy: 'train' (huấn luyện model) hoặc 'inference' (sinh ảnh ASCII)"
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/config.yaml', 
        help="Đường dẫn đến file cấu hình YAML (mặc định: configs/config.yaml)"
    )
    
    # 3. Đọc tham số người dùng nhập từ Terminal
    args = parser.parse_args()
    
    # 4. Load file cấu hình
    print(f"[*] Đang nạp cấu hình từ: {args.config}...")
    config = load_config(args.config)
    
    print("="*50)
    print(f" BẮT ĐẦU CHẠY DỰ ÁN: {config.get('project_name', 'ASCII-Generator')}")
    print("="*50)
    
    # 5. Điều hướng luồng chạy
    if args.mode == 'train':
        train_model(config)
    elif args.mode == 'inference':
        run_inference(config)

if __name__ == "__main__":
    main()