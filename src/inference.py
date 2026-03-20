# src/inference.py
import os
import torch
from PIL import Image
from torchvision import transforms
from src.model import ASCIICNN

# Danh sách ký tự ASCII (Cần khớp với lúc train, tạm thời dùng list chuẩn này)
ASCII_CHARS = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. " 

def load_trained_model(config, device):
    """Khởi tạo model và nạp trọng số đã train"""
    num_classes = config['model']['num_classes']
    model_path = os.path.join(config['model']['save_dir'], "ascii_cnn_model.pth")
    
    model = ASCIICNN(num_classes=num_classes).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Chuyển sang chế độ evaluation
        print(f">>> Đã load weights thành công từ: {model_path}")
    else:
        print(f"[Cảnh báo] Không tìm thấy file weights tại {model_path}.")
        
    return model

def image_to_ascii(image_path, model, config, device):
    """Biến đổi ảnh thật thành ASCII Art bằng AI và lưu ra file"""
    image_size = config['data']['image_size']
    
    try:
        img = Image.open(image_path).convert('L')
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy ảnh tại {image_path}")
        return

    width, height = img.size
    cols = width // image_size
    rows = height // image_size
    
    print(f"Kích thước lưới ASCII dự kiến: {cols} cột x {rows} hàng")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    ascii_art = []
    
    with torch.no_grad():
        for row in range(rows):
            ascii_row = ""
            for col in range(cols):
                box = (col * image_size, row * image_size, (col + 1) * image_size, (row + 1) * image_size)
                patch = img.crop(box)
                
                tensor_patch = transform(patch).unsqueeze(0).to(device)
                
                output = model(tensor_patch)
                _, predicted_idx = torch.max(output, 1)
                
                char_idx = predicted_idx.item()
                char_idx = min(char_idx, len(ASCII_CHARS) - 1) 
                
                ascii_row += ASCII_CHARS[char_idx]
            
            ascii_art.append(ascii_row)

    # ĐOẠN LƯU FILE NẰM Ở ĐÂY (Bên trong hàm image_to_ascii)
    output_txt_path = r"C:\Users\KHANH\Documents\GitHub\ASCII-generator-V2\data\processed\output_ascii.txt"
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True) 

    print("\n" + "="*50 + " KẾT QUẢ ASCII " + "="*50 + "\n")
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for row in ascii_art:
            print(row)
            f.write(row + "\n")
            
    print("\n" + "="*115)
    print(f">>> 🎉 THÀNH CÔNG! Đã lưu bản nét căng tại: {output_txt_path}")

def run_inference(config):
    print(">>> Đã vào luồng INFERENCE (Dự đoán ảnh thực tế)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_trained_model(config, device)
    
    test_image_path = r"C:\Users\KHANH\Documents\GitHub\ASCII-generator-V2\data\raw\test_image.jpg"
    print(f"Đang xử lý ảnh: {test_image_path}")
    
    image_to_ascii(test_image_path, model, config, device)