import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def train_single_epoch(model, data_loader, optimizer, device):
    """
    Train model for one epoch
    
    Args:
        model: The model to train
        data_loader: DataLoader for training data
        optimizer: Optimizer for parameter updates
        device: Device to train on (cpu/cuda)
        
    Returns:
        avg_loss: Average loss for the epoch
        accuracy: Accuracy for the epoch
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(data_loader, desc="Training")
    for batch_idx, (images, targets) in enumerate(progress_bar):
        # Move data to device
        images, targets = images.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits, loss = model(images, targets)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item() * images.size(0)
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)
        
        # Update progress bar
        accuracy = 100.0 * correct / total
        progress_bar.set_postfix({
            'loss': loss.item(),
            'acc': f"{accuracy:.2f}%"
        })
    
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy

@torch.no_grad()
def evaluate_model(model, data_loader, device):
    """
    Evaluate model on given data
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to evaluate on (cpu/cuda)
        
    Returns:
        avg_loss: Average loss for the evaluation
        accuracy: Accuracy for the evaluation
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, targets in tqdm(data_loader, desc="Evaluating"):
        # Move data to device
        images, targets = images.to(device), targets.to(device)
        
        # Forward pass
        logits, loss = model(images, targets)
        
        # Update metrics
        total_loss += loss.item() * images.size(0)
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)
    
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy

def train_model(model, train_loader, test_loader, num_epochs=5, learning_rate=0.001, 
               weight_decay=0.05, checkpoint_dir='checkpoints'):
    """
    Train the model for specified number of epochs
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        num_epochs: Number of epochs to train for
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        history: Dictionary containing training history
    """
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup device
    device = next(model.parameters()).device
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )
    
    # Initialize tracking variables
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'learning_rates': []
    }
    
    best_accuracy = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch
        train_loss, train_acc = train_single_epoch(
            model, train_loader, optimizer, device
        )
        
        # Evaluate on test set
        test_loss, test_acc = evaluate_model(
            model, test_loader, device
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['learning_rates'].append(current_lr)
        
        # Print epoch results
        print(f"  Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.2f}%")
        print(f"  Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.2f}%")
        print(f"  Learning rate: {current_lr:.6f}")
        
        # Save checkpoint if improved
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            
            # Create checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'test_loss': test_loss,
                'test_acc': test_acc
            }
            
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"  Saved new best model with accuracy: {test_acc:.2f}%")
    
    print(f"\nTraining completed! Best test accuracy: {best_accuracy:.2f}%")
    return history

def plot_training_history(history):
    """
    Plot training metrics
    
    Args:
        history: Dictionary containing training history
    """
    plt.figure(figsize=(12, 8))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot learning rate
    plt.subplot(2, 2, 3)
    plt.plot(history['learning_rates'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(alpha=0.3)
    
    # Plot improvement
    plt.subplot(2, 2, 4)
    plt.bar(['Initial', 'Final'], [history['test_acc'][0], history['test_acc'][-1]], color=['skyblue', 'navy'])
    plt.ylabel('Test Accuracy (%)')
    plt.title('Accuracy Improvement')
    
    # Add value labels
    for i, v in enumerate([history['test_acc'][0], history['test_acc'][-1]]):
        plt.text(i, v + 1, f"{v:.2f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

@torch.no_grad()
def get_predictions(model, data_loader, device):
    """
    Get model predictions on dataset
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to evaluate on (cpu/cuda)
        
    Returns:
        all_predictions: Model predictions
        all_targets: Ground truth labels
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    for images, targets in tqdm(data_loader, desc="Computing predictions"):
        # Move images to device
        images = images.to(device)
        
        # Get predictions
        logits, _ = model(images)
        predictions = torch.argmax(logits, dim=1)
        
        # Save predictions and targets
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(targets.numpy())
    
    return np.array(all_predictions), np.array(all_targets)

def analyze_results(predictions, targets, class_names):
    """
    Analyze model results
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
        class_names: List of class names
        
    Returns:
        results_dict: Dictionary with analysis results
    """
    # Compute confusion matrix
    cm = confusion_matrix(targets, predictions)
    
    # Separate MNIST and FashionMNIST results
    mnist_mask = targets < 10
    fashion_mask = targets >= 10
    
    mnist_acc = 100.0 * np.mean(predictions[mnist_mask] == targets[mnist_mask])
    fashion_acc = 100.0 * np.mean(predictions[fashion_mask] == targets[fashion_mask])
    overall_acc = 100.0 * np.mean(predictions == targets)
    
    # Create results dictionary
    results_dict = {
        'confusion_matrix': cm,
        'mnist_accuracy': mnist_acc,
        'fashion_accuracy': fashion_acc,
        'overall_accuracy': overall_acc
    }
    
    return results_dict

def plot_confusion_matrix(confusion_matrix, class_names):
    """
    Plot confusion matrix
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
    """
    plt.figure(figsize=(14, 12))
    
    # Plot confusion matrix with seaborn
    ax = sns.heatmap(
        confusion_matrix,
        annot=False,  # Don't show numbers in cells
        cmap='YlGnBu',  # Use a different colormap
        fmt='d',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    # Set labels and title
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=16)
    
    # Rotate tick labels
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def plot_dataset_comparison(mnist_acc, fashion_acc, overall_acc):
    """
    Plot comparison between MNIST and FashionMNIST performance
    
    Args:
        mnist_acc: Accuracy on MNIST subset
        fashion_acc: Accuracy on FashionMNIST subset
        overall_acc: Overall accuracy
    """
    plt.figure(figsize=(10, 6))
    
    # Create bar chart with different color
    bars = plt.bar(
        ['MNIST', 'FashionMNIST', 'Overall'],
        [mnist_acc, fashion_acc, overall_acc],
        color=['#ff9999', '#66b3ff', '#99ff99']
    )
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"{height:.2f}%",
            ha='center',
            fontsize=12
        )
    
    # Set labels and title
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Model Performance by Dataset', fontsize=16)
    
    # Set y-axis range
    plt.ylim(0, 105)
    
    # Add grid on y-axis
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('dataset_comparison.png')
    plt.show()

def visualize_predictions(model, dataset, num_samples=8, device='cpu'):
    """
    Visualize model predictions on random samples
    
    Args:
        model: Trained model
        dataset: Dataset to sample from
        num_samples: Number of samples to visualize
        device: Device to run inference on
    """
    model.eval()
    
    # Set up figure
    n_cols = 4
    n_rows = (num_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            # Get random sample
            idx = torch.randint(len(dataset), size=(1,)).item()
            img, label = dataset[idx]
            
            # Get prediction
            with torch.no_grad():
                logits, _ = model(img.unsqueeze(0).to(device))
                pred = torch.argmax(logits, dim=1).item()
            
            # Denormalize image
            img = img.squeeze().cpu()
            img = img * 0.5 + 0.5
            
            # Plot image
            ax.imshow(img, cmap='viridis')
            
            # Set title based on correctness
            title_color = 'green' if pred == label else 'red'
            ax.set_title(
                f"True: {dataset.category_names[label]}\n"
                f"Pred: {dataset.category_names[pred]}",
                color=title_color,
                fontsize=10
            )
            
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_examples.png')
    plt.show()