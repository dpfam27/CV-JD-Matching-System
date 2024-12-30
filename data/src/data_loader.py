# src/data/data_loader.py

import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os

class FashionDataset(Dataset):  # For Kaggle dataset
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Path to data/raw/fashion_dataset
            split (str): 'train' or 'test'
            transform: Image transformations
        """
        self.root_dir = os.path.join(root_dir, 'fashion_dataset', split)
        self.transform = transform
        self.classes = ['dresses', 'tops', 'bottoms']
        
        # Store image paths and labels
        self.images = []  # Will store: ['path/to/dress1.jpg', 'path/to/top1.jpg',...]
        self.labels = []  # Will store: [0, 1, 2, ...] (category indices)
        
        # Collect images from each category folder
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            # Loop through each image in category folder
            for img_name in os.listdir(class_dir):
                self.images.append(os.path.join(class_dir, img_name))
                self.labels.append(class_idx)
                
    def __len__(self):
        """Return total number of images"""
        return len(self.images)
        
    def __getitem__(self, idx):
        """
        Get single data item
        Args:
            idx (int): Index
        Returns:
            tuple: (image, label)
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Open and convert image to RGB
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, label

class SheinDataset(Dataset):  # For scraped Shein data
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Path to data/raw/shein_data
            split (str): 'train' or 'test'
            transform: Image transformations
        """
        self.root_dir = os.path.join(root_dir, 'shein_data', split)
        self.transform = transform
        
        # Read metadata CSV
        csv_path = os.path.join(self.root_dir, f'{split}.csv')
        self.data = pd.read_csv(csv_path)
        
    def __len__(self):
        """Return total number of images"""
        return len(self.data)
        
    def __getitem__(self, idx):
        """
        Get single data item
        Args:
            idx (int): Index
        Returns:
            dict: Contains image, category, and product URL
        """
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, 'images', row['local_path'])
        
        # Load and process image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'category': row['category'],
            'url': row['product_url']
        }