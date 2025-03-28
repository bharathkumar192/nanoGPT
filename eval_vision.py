"""
Evaluation script for VisionGPT.
This script loads a trained model and evaluates it on the test set.
It also includes visualization of model predictions.
"""

import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Import both model types
from vision_model import VisionGPTConfig, VisionGPT
from simple_vision_model import VisionGPTConfig as SimpleVisionGPTConfig
from simple_vision_model import SimpleVisionGPT
from data_utils import get_mnist_dataloaders

# MNIST and FashionMNIST class names
CLASS_NAMES = [
    # MNIST (0-9)
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    # FashionMNIST (10-19)
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained VisionGPT model')
    parser.add_argument('--checkpoint', type=str, default='out-vision-simple/ckpt.pt', help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='auto', choices=['auto', 'simple', 'rope'], 
                       help='Model type: auto (detect), simple (SimpleVisionGPT), or rope (VisionGPT)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--visualize', action='store_true', help='Visualize model predictions')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples to visualize')
    parser.add_argument('--save_dir', type=str, default='./visualizations', help='Directory to save visualizations')
    return parser.parse_args()

def detect_model_type(checkpoint_path):
    """Try to detect the model type from the checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model']
    
    # Check for RoPE-specific parameters
    has_rope = any('rope' in key for key in state_dict.keys())
    
    if has_rope:
        return 'rope'
    else:
        return 'simple'

def load_model(checkpoint_path, device, model_type='auto'):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint['model_args']
    
    # Detect model type if 'auto'
    if model_type == 'auto':
        model_type = detect_model_type(checkpoint_path)
    
    print(f"Using model type: {model_type}")
    
    # Create model with same config
    if model_type == 'rope':
        model_config = VisionGPTConfig(**model_args)
        model = VisionGPT(model_config)
    else:  # 'simple'
        model_config = SimpleVisionGPTConfig(**model_args)
        model = SimpleVisionGPT(model_config)
    
    # Load state dict
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    # Load model weights
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")
        print("Trying alternative model type...")
        
        # Try the other model type
        if model_type == 'rope':
            model_config = SimpleVisionGPTConfig(**model_args)
            model = SimpleVisionGPT(model_config)
        else:
            model_config = VisionGPTConfig(**model_args)
            model = VisionGPT(model_config)
        
        try:
            model.load_state_dict(state_dict)
            print("Successfully loaded with alternative model type.")
        except RuntimeError as e2:
            print(f"Failed with alternative model type as well: {e2}")
            raise RuntimeError("Could not load model with either model type. Check the checkpoint.")
    
    # Move to device and set to eval mode
    model.to(device)
    model.eval()
    
    return model

def evaluate_model(model, val_loader, device):
    """Evaluate model performance on validation/test set"""
    model.eval()
    all_preds = []
    all_targets = []
    running_loss = 0
    running_correct = 0
    total = 0
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass
            logits, loss = model(images, targets)
            
            # Get predictions
            _, preds = logits.max(1)
            
            # Update metrics
            running_loss += loss.item() * images.size(0)
            running_correct += preds.eq(targets).sum().item()
            total += targets.size(0)
            
            # Save predictions and targets for confusion matrix
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate final metrics
    avg_loss = running_loss / total
    accuracy = 100. * running_correct / total
    
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Display classification report
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=CLASS_NAMES))
    
    return all_preds, all_targets, avg_loss, accuracy

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix"""
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def visualize_predictions(model, val_loader, device, num_samples=16, save_dir=None):
    """Visualize model predictions on random samples"""
    model.eval()
    
    # Get batch of images
    dataiter = iter(val_loader)
    images, targets = next(dataiter)
    images, targets = images[:num_samples], targets[:num_samples]
    
    # Move to device
    images, targets = images.to(device), targets.to(device)
    
    # Get predictions
    with torch.no_grad():
        logits, _ = model(images)
        probs = torch.nn.functional.softmax(logits, dim=1)
        _, preds = logits.max(1)
    
    # Plot images with predictions
    fig = plt.figure(figsize=(20, 10))
    for i in range(num_samples):
        # Get image
        img = images[i].cpu().numpy().squeeze()
        true_label = targets[i].item()
        pred_label = preds[i].item()
        prob = probs[i, pred_label].item()
        
        # Determine if prediction is correct
        correct = true_label == pred_label
        color = 'green' if correct else 'red'
        
        # Plot image with prediction
        ax = fig.add_subplot(num_samples // 4, 4, i + 1)
        ax.imshow(img, cmap='gray')
        ax.set_title(f"True: {CLASS_NAMES[true_label]}\nPred: {CLASS_NAMES[pred_label]} ({prob:.2f})", 
                    color=color)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save figure if directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'predictions.png')
        plt.savefig(save_path)
        print(f"Predictions visualization saved to {save_path}")
    
    plt.show()

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Get dataloaders
    data = get_mnist_dataloaders(batch_size=args.batch_size)
    val_loader = data['val']
    
    # Load model
    model = load_model(args.checkpoint, device, model_type=args.model_type)
    
    # Evaluate model
    all_preds, all_targets, avg_loss, accuracy = evaluate_model(model, val_loader, device)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        all_targets, all_preds, CLASS_NAMES,
        save_path=os.path.join(args.save_dir, 'confusion_matrix.png')
    )
    
    # Visualize predictions if requested
    if args.visualize:
        visualize_predictions(model, val_loader, device, args.num_samples, args.save_dir)

if __name__ == "__main__":
    main()