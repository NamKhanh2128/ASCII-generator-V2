# src/utils.py
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

def set_seed(seed=42):
    """
    Cố định seed để đảm bảo kết quả train trên Kaggle 
    luôn đồng nhất qua các lần chạy (Reproducibility).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f">>> Đã cố định Random Seed: {seed}")

def plot_training_loss(train_losses, save_dir):
    """
    Vẽ và lưu biểu đồ quá trình giảm Loss sau khi train.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', color='blue', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Biểu đồ quá trình huấn luyện (Training Loss)')
    plt.grid(True)
    plt.legend()
    
    # Lưu file ảnh biểu đồ vào thư mục models/
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'loss_plot.png')
    plt.savefig(save_path)
    plt.close()
    print(f">>> Đã lưu biểu đồ Loss tại: {save_path}")