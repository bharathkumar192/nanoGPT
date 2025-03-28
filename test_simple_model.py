"""
Test script to validate the SimpleVisionGPT model before training.
This script runs a few forward passes to ensure all dimensions are correct.
"""

import torch
from simple_vision_model import VisionGPTConfig, SimpleVisionGPT
from data_utils import get_mnist_dataloaders

def test_model():
    """Test SimpleVisionGPT model"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define model config
    print("Creating model configuration...")
    config = VisionGPTConfig(
        img_size=28,
        patch_size=4,
        in_channels=1,
        n_layer=2,  # Small model for quick testing
        n_head=4,
        n_embd=128,
        n_classes=20,
        dropout=0.1,
        bias=True
    )
    
    # Print config details
    print(f"Model configuration:")
    print(f"  - Image size: {config.img_size}")
    print(f"  - Patch size: {config.patch_size}")
    print(f"  - Block size: {config.block_size}")
    print(f"  - Embed dim: {config.n_embd}")
    print(f"  - Head dim: {config.n_embd // config.n_head}")
    
    # Create model
    print("\nCreating SimpleVisionGPT model...")
    model = SimpleVisionGPT(config)
    model.to(device)
    print(f"Created model with {model.get_num_params()/1e6:.2f}M parameters")
    
    # Create a single batch
    print("\nGetting test batch...")
    dataloaders = get_mnist_dataloaders(batch_size=4, num_workers=0)
    images, labels = next(iter(dataloaders['train']))
    images = images.to(device)
    labels = labels.to(device)
    
    print(f"Input shape: {images.shape}")
    print(f"Labels: {labels}")
    
    # Test forward pass without targets (inference mode)
    print("\nTesting forward pass (inference)...")
    with torch.no_grad():
        logits, _ = model(images)
    
    print(f"Output logits shape: {logits.shape}")
    
    # Test the forward pass with targets (training mode)
    print("\nTesting forward pass (training)...")
    logits, loss = model(images, labels)
    print(f"Loss: {loss.item()}")
    
    # Get predictions
    _, predictions = logits.max(1)
    print(f"Predictions: {predictions}")
    
    # Test backward pass
    print("\nTesting backward pass...")
    loss.backward()
    
    print("All tests passed!")
    
    return model

if __name__ == "__main__":
    test_model()
