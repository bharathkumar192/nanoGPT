"""
Run script with instructions for training Vision-nanoGPT on MNIST/FashionMNIST.

This script contains instructions and examples for running the Vision-nanoGPT model.
"""

import os
import sys

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")

def print_command(cmd):
    """Print a formatted command"""
    print(f"  $ {cmd}")

def main():
    """Print instructions for running the Vision-nanoGPT model"""
    print_header("Vision-nanoGPT: Instructions")
    
    print("This project adapts nanoGPT for vision tasks, specifically for the combined")
    print("MNIST and FashionMNIST classification task (20 classes total).\n")
    
    print("We have implemented two versions:")
    print("1. VisionGPT: Using RoPE (Rotary Position Embeddings)")
    print("2. SimpleVisionGPT: Using standard positional embeddings\n")
    
    print("TESTING THE MODELS")
    print("-----------------")
    print("First, test that the models work with a small configuration:")
    print_command("python test_model.py")
    print_command("python test_simple_model.py")
    print("\n")
    
    print("TRAINING THE MODELS")
    print("------------------")
    print("Train the SimpleVisionGPT model (recommended for stability):")
    print_command("python train_simple_vision.py")
    print("\n")
    
    print("Train the VisionGPT model with RoPE:")
    print_command("python train_vision.py train_vision_no_compile.py")
    print("\n")
    
    print("EVALUATING THE MODELS")
    print("--------------------")
    print("After training, evaluate the model and visualize predictions:")
    print_command("python eval_vision.py --checkpoint out-vision-simple/ckpt.pt --visualize")
    print("\n")
    
    print("PROJECT STRUCTURE")
    print("----------------")
    print("- vision_model.py: VisionGPT model with RoPE")
    print("- simple_vision_model.py: SimpleVisionGPT model without RoPE")
    print("- data_utils.py: Utilities for loading and processing MNIST/FashionMNIST data")
    print("- train_vision.py: Training script for VisionGPT")
    print("- train_simple_vision.py: Training script for SimpleVisionGPT")
    print("- eval_vision.py: Evaluation and visualization script")
    print("- test_model.py: Test script for VisionGPT")
    print("- test_simple_model.py: Test script for SimpleVisionGPT")
    print("\n")
    
    print("METHODOLOGY DETAILS")
    print("-----------------")
    print("This implementation adapts nanoGPT for vision tasks by:")
    print("1. Replacing token embeddings with image patch embeddings (like ViT)")
    print("2. Using a classification head instead of a language modeling head")
    print("3. Removing causal masking in self-attention")
    print("4. Incorporating RoPE (in the VisionGPT version) for better spatial understanding")
    print("5. Using the CLS token approach for classification (like in ViT)")
    print("\n")
    
    print("The implementation borrows ideas from the I-JEPA paper, particularly:")
    print("- The patch-based image representation")
    print("- The focus on learning semantic features")
    print("- Efficient computational approach")
    print("\n")
    
    print_header("Ready to Run!")

if __name__ == "__main__":
    main()
