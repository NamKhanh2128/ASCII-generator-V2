# src/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import get_dataloaders
from src.model import ASCIICNN
from src.utils import set_seed, plot_training_loss

def train_model(config):
    # 1. Cố định Random Seed để đảm bảo tính tái lập (Reproducibility)
    set_seed(42)

    # 2. Setup Device (Nhận diện GPU nếu khả dụng)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Đang sử dụng thiết bị: {device}")

    # 3. Chuẩn bị DataLoader cho Train và Validation
    train_dir = config['data']['processed_dir']
    batch_size = config['model']['batch_size']
    image_size = config['data']['image_size']
    val_split = config['data'].get('val_split', 0.2)
    
    # Bỏ qua bước train nếu thư mục không tồn tại (hỗ trợ chạy giả lập/test luồng)
    if not os.path.exists(train_dir) or not os.listdir(train_dir):
        print(f"[Cảnh báo] Không tìm thấy dữ liệu tại {train_dir} hoặc thư mục rỗng. Đang chạy giả lập...")
        return
        
    train_loader, val_loader = get_dataloaders(
        data_dir=train_dir, 
        batch_size=batch_size, 
        image_size=image_size, 
        val_split=val_split,
        seed=42
    )

    if train_loader is None or len(train_loader) == 0:
        print("[LỖI] Dữ liệu huấn luyện rỗng. Vui lòng kiểm tra lại data!")
        return

    # 4. Khởi tạo Model, Loss và Optimizer
    model = ASCIICNN(num_classes=config['model']['num_classes']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['model']['learning_rate'])

    # 5. Vòng lặp Training & Validation
    epochs = config['model']['epochs']
    print(f">>> BẮT ĐẦU TRAINING ({epochs} Epochs, Tỉ lệ Validation: {val_split})...")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    save_dir = config['model']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "ascii_cnn_model.pth")
    last_model_path = os.path.join(save_dir, "ascii_cnn_last_model.pth")

    for epoch in range(epochs):
        # --- PHASE TRAINING ---
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # --- PHASE VALIDATION ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / max(len(val_loader), 1)
        val_losses.append(epoch_val_loss)
        val_acc = 100.0 * correct / max(total, 1)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Lưu checkpoint tốt nhất dựa trên Validation Loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"   => 🎉 Đã lưu model tốt nhất mới tại: {best_model_path}")

    # Lưu model ở epoch cuối cùng
    torch.save(model.state_dict(), last_model_path)
    print(f">>> Hoàn tất Training! Model cuối cùng được lưu tại: {last_model_path}")

    # 6. Vẽ biểu đồ hiển thị cả Train & Val Loss
    plot_training_loss(train_losses, save_dir, val_losses)