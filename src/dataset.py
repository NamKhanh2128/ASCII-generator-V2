# src/dataset.py
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ASCIIDataset(Dataset):
    def __init__(self, data_dir=None, image_size=32, is_train=True, image_paths=None, labels=None):
        self.image_size = image_size
        self.is_train = is_train
        
        if image_paths is not None and labels is not None:
            self.image_paths = image_paths
            self.labels = labels
        else:
            self.image_paths = []
            self.labels = []
            
            if data_dir and os.path.exists(data_dir):
                for label_name in os.listdir(data_dir):
                    label_dir = os.path.join(data_dir, label_name)
                    if os.path.isdir(label_dir):
                        for img_name in os.listdir(label_dir):
                            self.image_paths.append(os.path.join(label_dir, img_name))
                            try:
                                self.labels.append(int(label_name))
                            except ValueError:
                                # Phòng trường hợp thư mục có tên không phải dạng số nguyên
                                continue

        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=1), 
                # Có thể thêm Data Augmentation ở đây trong tương lai
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

def get_dataloaders(data_dir, batch_size, image_size, val_split=0.2, seed=42):
    """Đọc toàn bộ thư mục dữ liệu, chia ngẫu nhiên thành train và validation loaders."""
    full_dataset = ASCIIDataset(data_dir=data_dir, image_size=image_size, is_train=True)
    
    total_len = len(full_dataset)
    if total_len == 0:
        return None, None
        
    indices = list(range(total_len))
    rng = random.Random(seed)
    rng.shuffle(indices)
    
    val_size = int(total_len * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_paths = [full_dataset.image_paths[i] for i in train_indices]
    train_labels = [full_dataset.labels[i] for i in train_indices]
    
    val_paths = [full_dataset.image_paths[i] for i in val_indices]
    val_labels = [full_dataset.labels[i] for i in val_indices]
    
    train_dataset = ASCIIDataset(image_size=image_size, is_train=True, image_paths=train_paths, labels=train_labels)
    val_dataset = ASCIIDataset(image_size=image_size, is_train=False, image_paths=val_paths, labels=val_labels)
    
    # Tối ưu cho hệ điều hành Windows: num_workers=0 chạy an toàn và nhanh hơn
    num_workers = 0 if os.name == 'nt' else 2
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader

def get_dataloader(data_dir, batch_size, image_size, is_train=True):
    """Hàm cũ phục vụ tính tương thích ngược."""
    num_workers = 0 if os.name == 'nt' else 2
    dataset = ASCIIDataset(data_dir=data_dir, image_size=image_size, is_train=is_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers, pin_memory=True)
    return dataloader