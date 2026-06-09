# tests/test_utils.py
import os
import torch
import tempfile
from src.utils import set_seed, plot_training_loss

def test_set_seed():
    """Kiểm tra xem hàm set_seed có thực sự cố định được tính ngẫu nhiên không"""
    # Cố định seed lần 1 và sinh ra một tensor ngẫu nhiên
    set_seed(42)
    tensor1 = torch.rand(3, 3)
    
    # Cố định lại đúng seed đó và sinh ra tensor thứ 2
    set_seed(42)
    tensor2 = torch.rand(3, 3)
    
    # Nếu hàm hoạt động đúng, 2 tensor phải giống hệt nhau 100%
    assert torch.equal(tensor1, tensor2), "LỖI: set_seed không hoạt động, 2 lần gọi cho ra kết quả khác nhau!"
    print("✔ [PASS] test_set_seed: Tính ngẫu nhiên đã được kiểm soát tốt.")

def test_plot_training_loss():
    """Kiểm tra xem hàm vẽ biểu đồ có sinh ra file ảnh .png thành công không"""
    # Dữ liệu loss giả lập giảm dần
    fake_losses = [0.9, 0.7, 0.5, 0.3, 0.1]
    fake_val_losses = [0.95, 0.8, 0.6, 0.45, 0.35]
    
    # Tạo một thư mục tạm thời (chạy xong tự xóa để không làm bẩn project)
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test 1: Chỉ vẽ Train Loss
        plot_training_loss(fake_losses, temp_dir)
        expected_file_path = os.path.join(temp_dir, 'loss_plot.png')
        assert os.path.exists(expected_file_path), "LỖI: plot_training_loss không tạo được file ảnh!"
        os.remove(expected_file_path) # Xóa để test lần 2
        
        # Test 2: Vẽ cả Train Loss và Val Loss
        plot_training_loss(fake_losses, temp_dir, fake_val_losses)
        assert os.path.exists(expected_file_path), "LỖI: plot_training_loss khi có Val Loss không tạo được file ảnh!"
        
        print("✔ [PASS] test_plot_training_loss: Vẽ biểu đồ Loss đơn và kép thành công.")

if __name__ == "__main__":
    print("="*40)
    print(">>> BẮT ĐẦU CHẠY UNIT TEST...")
    print("="*40)
    
    test_set_seed()
    test_plot_training_loss()
    
    print("="*40)
    print(">>> TOÀN BỘ TEST HOÀN TẤT VÀ THÀNH CÔNG! BẠN CÓ THỂ YÊN TÂM CODE TIẾP.")