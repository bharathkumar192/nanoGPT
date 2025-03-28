"""
Test script to validate the VisionGPT model before training.
This script runs a few forward passes to ensure all dimensions are correct.
"""

import torch
from vision_model import VisionGPTConfig, VisionGPT
from data_utils import get_mnist_dataloaders

def test_model():
    """Test VisionGPT model"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define model config
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
    
    # Create model
    model = VisionGPT(config)
    model.to(device)
    print(f"Created model with {model.get_num_params()/1e6:.2f}M parameters")
    
    # Print the model
    print("Model summary:")
    print(model)
    
    # Create a single batch
    print("Getting test batch...")
    dataloaders = get_mnist_dataloaders(batch_size=4, num_workers=0)
    images, labels = next(iter(dataloaders['train']))
    images = images.to(device)
    labels = labels.to(device)
    
    print(f"Input shape: {images.shape}")
    print(f"Labels: {labels}")
    
    # Test forward pass without targets (inference mode)
    print("Testing forward pass (inference)...")
    with torch.no_grad():
        logits, _ = model(images)
    
    print(f"Output logits shape: {logits.shape}")
    
    # Test the forward pass with targets (training mode)
    print("Testing forward pass (training)...")
    logits, loss = model(images, labels)
    print(f"Loss: {loss.item()}")
    
    # Get predictions
    _, predictions = logits.max(1)
    print(f"Predictions: {predictions}")
    
    # Test backward pass
    print("Testing backward pass...")
    loss.backward()
    
    print("All tests passed!")
    
    # Additional test for RoPE implementation
    print("\nTesting RoPE implementation...")
    from vision_model import RotaryEmbedding
    
    # Create a small tensor for testing
    batch_size = 2
    n_head = 2
    seq_len = 5
    head_dim = 8
    
    # Create RoPE module
    rope = RotaryEmbedding(dim=head_dim, max_seq_len=seq_len)
    rope.to(device)
    
    # Create random tensor
    x = torch.randn(batch_size, n_head, seq_len, head_dim, device=device)
    print(f"Input shape to RoPE: {x.shape}")
    
    # Apply RoPE
    x_rot = rope(x)
    print(f"Output shape from RoPE: {x_rot.shape}")
    
    print("RoPE test passed!")
    
    return model

if __name__ == "__main__":
    test_model()
