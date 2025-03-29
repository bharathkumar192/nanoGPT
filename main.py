import torch
import numpy as np
import os
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(456)
np.random.seed(456)

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Import from our modules
from dataset_class import create_data_loaders, DigitFashionDataset, get_transforms
from vision_transformer import ImageTransformer
from training_utils import (
    train_model, 
    plot_training_history, 
    get_predictions,
    analyze_results,
    plot_confusion_matrix,
    plot_dataset_comparison,
    visualize_predictions
)

# Path for saving checkpoints
CHECKPOINT_DIR = 'model_checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Set hyperparameters
BATCH_SIZE = 128
IMAGE_SIZE = 28
PATCH_SIZE = 4
INPUT_CHANNELS = 1
NUM_CLASSES = 20
EMBED_SIZE = 160
NUM_LAYERS = 6
NUM_HEADS = 5
EXPANSION_FACTOR = 4
DROPOUT = 0.1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.05
NUM_EPOCHS = 5

def main():
    """Main function to run training and evaluation"""
    # Create data loaders
    train_loader, test_loader, class_names = create_data_loaders(
        batch_size=BATCH_SIZE,
        num_workers=2
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Number of classes: {len(class_names)}")
    
    # Create model
    model = ImageTransformer(
        img_resolution=IMAGE_SIZE,
        patch_dimension=PATCH_SIZE,
        input_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        embed_size=EMBED_SIZE,
        num_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        mlp_expansion_factor=EXPANSION_FACTOR,
        dropout_prob=DROPOUT
    ).to(device)
    
    # Train model
    print("\nStarting training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        checkpoint_dir=CHECKPOINT_DIR
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Load best model for evaluation
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']+1} with accuracy {checkpoint['test_acc']:.2f}%")
    
    # Get predictions for analysis
    test_dataset = DigitFashionDataset(is_training=False, transform_fn=get_transforms())
    predictions, targets = get_predictions(model, test_loader, device)
    
    # Analyze results
    results = analyze_results(predictions, targets, class_names)
    
    # Plot confusion matrix
    plot_confusion_matrix(results['confusion_matrix'], class_names)
    
    # Plot dataset comparison
    plot_dataset_comparison(
        results['mnist_accuracy'],
        results['fashion_accuracy'],
        results['overall_accuracy']
    )
    
    # Visualize predictions
    visualize_predictions(model, test_dataset, num_samples=8, device=device)
    
    print("\nAnalysis completed!")
    print(f"MNIST accuracy: {results['mnist_accuracy']:.2f}%")
    print(f"FashionMNIST accuracy: {results['fashion_accuracy']:.2f}%")
    print(f"Overall accuracy: {results['overall_accuracy']:.2f}%")

if __name__ == "__main__":
    main()