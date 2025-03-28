"""
Dataset utilities for vision tasks.
This module provides functionality for loading and preprocessing MNIST and FashionMNIST datasets.
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms

class CombinedMNISTDataset(Dataset):
    """
    Dataset that combines MNIST and FashionMNIST datasets.
    MNIST labels are 0-9, FashionMNIST labels are shifted to 10-19.
    """
    
    def __init__(self, root='./data', train=True, transform=None, download=True):
        """
        Initialize the combined dataset.
        
        Args:
            root (str): Directory to store the dataset
            train (bool): If True, use the training set, otherwise use test set
            transform (callable): Optional transform to be applied on a sample
            download (bool): If True, download the dataset if not already downloaded
        """
        self.transform = transform or transforms.ToTensor()
        
        # Get MNIST
        mnist = datasets.MNIST(root=root, train=train, download=download, transform=None)
        
        # Get FashionMNIST
        fashion_mnist = datasets.FashionMNIST(root=root, train=train, download=download, transform=None)
        
        # Convert data to appropriate format
        mnist_data = mnist.data.unsqueeze(1).float() / 255.0  # Add channel dimension and normalize
        fashion_data = fashion_mnist.data.unsqueeze(1).float() / 255.0
        
        # Combine data
        self.data = torch.cat([mnist_data, fashion_data])
        
        # Combine targets, shifting FashionMNIST by 10
        self.targets = torch.cat([
            mnist.targets,
            fashion_mnist.targets + 10
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, target = self.data[idx], int(self.targets[idx])
        
        # Apply transforms if specified
        if self.transform is not None:
            # Convert to PIL for transforms
            img_np = img.squeeze().numpy() * 255
            img_np = img_np.astype(np.uint8)
            img_pil = Image.fromarray(img_np, mode='L')
            img = self.transform(img_pil)
        
        return img, target

def get_mnist_transforms(img_size=28):
    """
    Get transforms for MNIST/FashionMNIST datasets.
    
    Args:
        img_size (int): Target image size
    
    Returns:
        dict: Dictionary with train and val transforms
    """
    # For training: add some augmentations
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # For validation/testing: just normalize
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    return {
        'train': train_transform,
        'val': val_transform
    }

def get_mnist_dataloaders(batch_size, num_workers=4):
    """
    Create dataloaders for the combined MNIST/FashionMNIST dataset.
    
    Args:
        batch_size (int): Batch size for the dataloaders
        num_workers (int): Number of worker processes for data loading
    
    Returns:
        dict: Dictionary containing train and val dataloaders
    """
    # Get transforms
    transforms_dict = get_mnist_transforms()
    
    # Create datasets
    train_dataset = CombinedMNISTDataset(
        root='./data',
        train=True,
        transform=transforms_dict['train'],
        download=True
    )
    
    val_dataset = CombinedMNISTDataset(
        root='./data',
        train=False,
        transform=transforms_dict['val'],
        download=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create a metadata file for compatibility with nanoGPT
    meta = {
        'vocab_size': 20,  # 20 classes total (10 from MNIST + 10 from FashionMNIST)
    }
    
    os.makedirs('./data/mnist_fashion_combined', exist_ok=True)
    with open('./data/mnist_fashion_combined/meta.pkl', 'wb') as f:
        pickle.dump(meta, f)
    
    return {
        'train': train_loader,
        'val': val_loader,
        'meta': meta
    }

# For batch generation that mimics the nanoGPT interface
class ImageBatchGenerator:
    """
    A generator that provides batches of images in a format compatible with the nanoGPT training loop.
    This allows us to use the existing training infrastructure of nanoGPT.
    """
    
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
    
    def get_batch(self, split):
        """
        Get a batch of images and targets.
        
        Args:
            split (str): 'train' or 'val' (unused, kept for compatibility)
        
        Returns:
            tuple: (images, targets)
        """
        try:
            batch = next(self.iterator)
        except StopIteration:
            # Restart iterator if it's exhausted
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        
        images, targets = batch
        return images, targets