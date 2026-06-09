# src/inference.py
import os
import torch
from PIL import Image
from torchvision import transforms
from src.model import ASCIICNN

# Danh sách ký tự ASCII (Cần khớp với lúc train, tạm thời dùng list chuẩn này)
ASCII_CHARS = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\\\|()1{}[]?-_+~<>i!lI;:,\"^`'. " 

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

def image_to_ascii(image_path, output_txt_path, model, config, device):
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

    # Tối ưu hóa: Thu thập tất cả các patch ảnh để inference theo batch
    patches = []
    for row in range(rows):
        for col in range(cols):
            box = (col * image_size, row * image_size, (col + 1) * image_size, (row + 1) * image_size)
            patch = img.crop(box)
            patches.append(transform(patch))

    if not patches:
        print("[Cảnh báo] Lưới ASCII rỗng. Ảnh có thể quá nhỏ so với patch size.")
        return

    # Gom các patch thành tensor đơn (N, 1, H, W)
    all_patches_tensor = torch.stack(patches).to(device)
    
    # Chia theo batch nhỏ để inference tránh quá tải RAM/VRAM
    batch_size = 256
    num_patches = all_patches_tensor.size(0)
    predicted_indices = []
    
    with torch.no_grad():
        for i in range(0, num_patches, batch_size):
            batch_tensor = all_patches_tensor[i : i + batch_size]
            output = model(batch_tensor)
            _, predicted_batch = torch.max(output, 1)
            predicted_indices.extend(predicted_batch.cpu().tolist())

    # Ánh xạ kết quả về cấu trúc lưới ASCII
    ascii_art = []
    idx = 0
    for row in range(rows):
        ascii_row = ""
        for col in range(cols):
            char_idx = predicted_indices[idx]
            char_idx = min(char_idx, len(ASCII_CHARS) - 1) 
            ascii_row += ASCII_CHARS[char_idx]
            idx += 1
        ascii_art.append(ascii_row)

    # Lưu kết quả
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
    
    test_image_path = config['data'].get('input_image_path', 'data/raw/test_image.jpg')
    output_txt_path = config['data'].get('output_txt_path', 'data/processed/output_ascii.txt')
    
    print(f"Đang xử lý ảnh: {test_image_path}")
    
    image_to_ascii(test_image_path, output_txt_path, model, config, device)