# src/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import get_dataloader
from src.model import ASCIICNN

def train_model(config):
    # 1. Setup Device (Tự động nhận diện GPU trên Kaggle)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Đang sử dụng thiết bị: {device}")

    # 2. Chuẩn bị DataLoader
    train_dir = config['data']['processed_dir']
    batch_size = config['model']['batch_size']
    image_size = config['data']['image_size']
    
    # Bỏ qua bước train nếu thư mục không tồn tại (hỗ trợ test luồng)
    if not os.path.exists(train_dir):
        print(f"[Cảnh báo] Không tìm thấy dữ liệu tại {train_dir}. Đang chạy giả lập...")
        return
        
    train_loader = get_dataloader(train_dir, batch_size, image_size, is_train=True)

    # 3. Khởi tạo Model, Loss và Optimizer
    model = ASCIICNN(num_classes=config['model']['num_classes']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['model']['learning_rate'])

    # 4. Vòng lặp Training
    epochs = config['model']['epochs']
    print(">>> BẮT ĐẦU TRAINING...")
    
    for epoch in range(epochs):
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
            
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}")

    # 5. Lưu model (Kaggle sẽ lưu vào /kaggle/working/ theo config)
    save_dir = config['model']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "ascii_cnn_model.pth")
    
    torch.save(model.state_dict(), save_path)
    print(f">>> Hoàn tất Training! Model weight được lưu tại: {save_path}")