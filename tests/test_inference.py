# tests/test_inference.py
import os
import torch
import tempfile
from PIL import Image
from src.model import ASCIICNN
from src.inference import image_to_ascii, load_trained_model, ASCII_CHARS

def test_load_trained_model():
    """Kiểm tra xem load_trained_model có tạo được model ASCIICNN không."""
    config = {
        'model': {
            'num_classes': 70,
            'save_dir': 'non_existent_directory' # Sẽ in cảnh báo nhưng vẫn tạo được model rỗng
        }
    }
    device = torch.device('cpu')
    model = load_trained_model(config, device)
    
    assert isinstance(model, ASCIICNN), "LỖI: load_trained_model không trả về đối tượng ASCIICNN!"
    print("✔ [PASS] test_load_trained_model: Khởi tạo model thành công.")

def test_batch_inference_consistency():
    """Kiểm tra độ chính xác và tính tương đồng của cơ chế Batch Inference."""
    # 1. Cấu hình giả lập
    image_size = 32
    num_classes = 70
    config = {
        'data': {
            'image_size': image_size
        },
        'model': {
            'num_classes': num_classes,
            'save_dir': 'models/'
        }
    }
    
    # 2. Tạo một ảnh giả kích thước 96x64 (3 cột x 2 hàng patch size 32)
    width = image_size * 3
    height = image_size * 2
    img = Image.new('L', (width, height), color=128)
    
    # Tạo một số họa tiết khác nhau cho các patch
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    # Vẽ vài hình tròn/vuông để các patch không giống hệt nhau hoàn toàn
    draw.rectangle([0, 0, 32, 32], fill=50)
    draw.ellipse([32, 0, 64, 32], fill=200)
    draw.rectangle([64, 32, 96, 64], fill=250)
    
    # 3. Tạo một model ngẫu nhiên
    device = torch.device('cpu')
    model = ASCIICNN(num_classes=num_classes).to(device)
    model.eval()
    
    # 4. Lưu ảnh giả lập ra file tạm để chạy test (dùng PNG không nén hao hụt)
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = os.path.join(temp_dir, 'test_img.png')
        img.save(image_path)
        
        output_txt_path = os.path.join(temp_dir, 'output_ascii.txt')
        
        # 5. Chạy Batch Inference mới
        image_to_ascii(image_path, output_txt_path, model, config, device)
        
        # 6. Kiểm tra xem file kết quả có được tạo ra không
        assert os.path.exists(output_txt_path), "LỖI: Không tạo được file output ASCII!"
        
        # Đọc nội dung file kết quả
        with open(output_txt_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            
        # Kiểm tra cấu trúc lưới kết quả (phải là 2 hàng và mỗi hàng 3 ký tự)
        assert len(lines) == 2, f"LỖI: Số hàng không khớp. Mong đợi 2, nhận được {len(lines)}"
        for line in lines:
            assert len(line) == 3, f"LỖI: Số cột không khớp. Mong đợi 3, nhận được {len(line)}"
            
        # 7. So sánh tính nhất quán giữa Batch Inference và Single Inference (đọc cùng một ảnh từ disk)
        ref_img = Image.open(image_path).convert('L')
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        expected_ascii = []
        with torch.no_grad():
            for row in range(2):
                ascii_row = ""
                for col in range(3):
                    box = (col * image_size, row * image_size, (col + 1) * image_size, (row + 1) * image_size)
                    patch = ref_img.crop(box)
                    tensor_patch = transform(patch).unsqueeze(0).to(device)
                    output = model(tensor_patch)
                    _, predicted_idx = torch.max(output, 1)
                    char_idx = predicted_idx.item()
                    char_idx = min(char_idx, len(ASCII_CHARS) - 1)
                    ascii_row += ASCII_CHARS[char_idx]
                expected_ascii.append(ascii_row)
                
        # So sánh 2 kết quả
        assert lines == expected_ascii, f"LỖI: Kết quả Batch Inference ({lines}) khác biệt so với Single Inference ({expected_ascii})!"
        
        print("✔ [PASS] test_batch_inference_consistency: Thuật toán Batch Inference cho kết quả hoàn toàn chính xác so với tuần tự.")

if __name__ == "__main__":
    print("="*40)
    print(">>> BẮT ĐẦU CHẠY INFERENCE UNIT TEST...")
    print("="*40)
    test_load_trained_model()
    test_batch_inference_consistency()
    print("="*40)
    print(">>> INFERENCE UNIT TEST HOÀN TẤT VÀ THÀNH CÔNG!")
