# src/dataset.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ASCIIDataset(Dataset):
    def __init__(self, data_dir, image_size, is_train=True):
        self.data_dir = data_dir
        self.image_paths = []
        self.labels = []
        
        if os.path.exists(data_dir):
            for label_name in os.listdir(data_dir):
                label_dir = os.path.join(data_dir, label_name)
                if os.path.isdir(label_dir):
                    for img_name in os.listdir(label_dir):
                        self.image_paths.append(os.path.join(label_dir, img_name))
                        self.labels.append(int(label_name)) 

        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=1), 
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

def get_dataloader(data_dir, batch_size, image_size, is_train=True):
    dataset = ASCIIDataset(data_dir, image_size, is_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=2, pin_memory=True)
    return dataloader