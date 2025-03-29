import torch
import torch.utils.data
from torchvision import datasets, transforms

class DigitFashionDataset(torch.utils.data.Dataset):
    """
    Custom dataset combining MNIST and FashionMNIST datasets
    """
    def __init__(self, is_training=True, transform_fn=None):
        # Load digit dataset
        digit_data = datasets.MNIST(
            root='./dataset_files',
            train=is_training,
            download=True,
            transform=None
        )
        
        # Load fashion dataset
        fashion_data = datasets.FashionMNIST(
            root='./dataset_files',
            train=is_training,
            download=True,
            transform=None
        )
        
        # Combine images (normalize to [0-1] range)
        self.img_tensor = torch.cat([
            digit_data.data.float() / 255.0,
            fashion_data.data.float() / 255.0
        ]).unsqueeze(1)  # Add channel dimension
        
        # Combine targets (offset fashion by 10)
        self.target_tensor = torch.cat([
            digit_data.targets,
            fashion_data.targets + 10
        ])
        
        # Set up class names
        self.category_names = [
            # MNIST digit class names
            'Digit-0', 'Digit-1', 'Digit-2', 'Digit-3', 'Digit-4',
            'Digit-5', 'Digit-6', 'Digit-7', 'Digit-8', 'Digit-9',
            # FashionMNIST class names
            'T-shirt', 'Trousers', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
        
        self.transform_fn = transform_fn
    
    def __len__(self):
        return len(self.img_tensor)
    
    def __getitem__(self, idx):
        img = self.img_tensor[idx]
        
        if self.transform_fn:
            img = self.transform_fn(img)
            
        return img, self.target_tensor[idx]

# Create data transformation
def get_transforms():
    """Define transformations for training and testing"""
    # Standardize to [-1, 1] range
    base_transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    return base_transform

# Create data loaders function
def create_data_loaders(batch_size=128, num_workers=2):
    """Create and return data loaders for training and testing"""
    # Get transformations
    transform = get_transforms()
    
    # Create datasets
    train_dataset = DigitFashionDataset(is_training=True, transform_fn=transform)
    test_dataset = DigitFashionDataset(is_training=False, transform_fn=transform)
    
    # Create loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset.category_names