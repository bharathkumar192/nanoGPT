"""
Training script for VisionGPT.
This script adapts the nanoGPT training loop for vision tasks,
specifically for the combined MNIST/FashionMNIST classification task.
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from vision_model import VisionGPTConfig, VisionGPT
from data_utils import get_mnist_dataloaders

# -----------------------------------------------------------------------------
# default config values
# I/O
out_dir = 'out-vision'
eval_interval = 200
log_interval = 10
eval_iters = 20
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# wandb logging
wandb_log = False
wandb_project = 'vision-gpt'
wandb_run_name = 'vision-gpt'

# data
batch_size = 64
gradient_accumulation_steps = 1

# model
n_layer = 8
n_head = 8
n_embd = 256
dropout = 0.1
bias = True
img_size = 28
patch_size = 4
in_channels = 1
n_classes = 20  # 10 from MNIST + 10 from FashionMNIST

# adamw optimizer
learning_rate = 3e-4
max_iters = 5000
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 500
lr_decay_iters = 5000
min_lr = 3e-5

# DDP settings
backend = 'nccl'

# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True if torch.cuda.is_available() else False

# -----------------------------------------------------------------------------
# Let's allow command line overrides or a separate config file
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
try:
    exec(open('configurator.py').read())
except FileNotFoundError:
    pass

config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# Initialize distributed training if needed
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    # Scale down gradient accumulation steps for DDP
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # Single process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

# Create output directory
if master_process:
    os.makedirs(out_dir, exist_ok=True)

# Set random seed
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Set device and datatype
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Get data loaders
data = get_mnist_dataloaders(batch_size=batch_size)
train_loader = data['train']
val_loader = data['val']

# Initialize model
iter_num = 0
best_val_loss = float('inf')

def create_model():
    # Initialize a new VisionGPT model
    config_args = dict(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        n_classes=n_classes,
        dropout=dropout,
        bias=bias,
    )
    model_config = VisionGPTConfig(**config_args)
    model = VisionGPT(model_config)
    return model, config_args

if init_from == 'scratch':
    # Initialize new model
    model, model_args = create_model()
elif init_from == 'resume':
    # Resume from checkpoint
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint['model_args']
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    
    # Create model with same configuration
    model_config = VisionGPTConfig(**model_args)
    model = VisionGPT(model_config)
    
    # Load state dict
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

# Move model to device
model.to(device)

# Initialize optimizer
optimizer = model.configure_optimizers(
    weight_decay=weight_decay,
    learning_rate=learning_rate,
    betas=(beta1, beta2),
    device_type=device_type
)

# Resume optimizer state if continuing from checkpoint
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

# Initialize gradient scaler for mixed precision training
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Compile model if PyTorch 2.0 is available and compile flag is set
if compile:
    print("Compiling the model (this may take a minute)...")
    try:
        model = torch.compile(model)
    except Exception as e:
        print(f"Model compilation failed with error: {e}")
        print("Continuing with uncompiled model...")

# Setup DDP if needed
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Function to evaluate model performance
@torch.no_grad()
def evaluate():
    model.eval()
    losses = torch.zeros(eval_iters)
    accuracies = torch.zeros(eval_iters)
    
    for k in range(eval_iters):
        # Get batch from validation loader
        is_last_batch = k == eval_iters - 1
        batch_idx = k % len(val_loader)
        data_iter = iter(val_loader)
        for _ in range(batch_idx + 1):
            x, y = next(data_iter)
        
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        with ctx:
            logits, loss = model(x, y)
        
        # Compute accuracy
        _, predicted = logits.max(1)
        correct = predicted.eq(y).sum().item()
        accuracy = 100. * correct / y.size(0)
        
        losses[k] = loss.item()
        accuracies[k] = accuracy
    
    model.train()
    return losses.mean().item(), accuracies.mean().item()

# Learning rate schedule (with warmup)
def get_lr(it):
    # Linear warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # Cosine decay after warmup
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# Initialize wandb logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# Training loop
model.train()
train_iter = iter(train_loader)
t0 = time.time()
local_iter_num = 0
running_loss = 0
running_accuracy = 0

print(f"Starting training for {max_iters} iterations on {device}")

while True:
    # Check if we've reached max iterations
    if iter_num >= max_iters:
        break
    
    # Update learning rate according to schedule
    if decay_lr:
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Evaluate the model periodically
    if iter_num % eval_interval == 0 and master_process:
        val_loss, val_acc = evaluate()
        print(f"Step {iter_num}: val loss {val_loss:.4f}, val accuracy {val_acc:.2f}%")
        
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "lr": lr,
            })
        
        # Save checkpoint for best model
        if val_loss < best_val_loss or always_save_checkpoint:
            best_val_loss = val_loss
            if iter_num > 0:
                checkpoint = {
                    'model': model.module.state_dict() if ddp else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"Saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    
    # Exit if only evaluating
    if eval_only:
        break
    
    # Gradient accumulation loop
    for micro_step in range(gradient_accumulation_steps):
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        x, y = batch
        x, y = x.to(device), y.to(device)
        
        # Forward and backward pass
        with ctx:
            logits, loss = model(x, y)
            loss = loss / gradient_accumulation_steps  # Scale loss for gradient accumulation
        
        # Compute accuracy
        _, predicted = logits.max(1)
        correct = predicted.eq(y).sum().item()
        accuracy = 100. * correct / y.size(0)
        
        # Update running metrics
        running_loss += loss.item() * gradient_accumulation_steps
        running_accuracy += accuracy
        
        # Backward pass with gradient scaling for fp16
        scaler.scale(loss).backward()
    
    # Clip gradients if needed
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # Step optimizer and scaler
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    
    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    
    if iter_num % log_interval == 0 and master_process:
        # Log training metrics
        avg_loss = running_loss / log_interval
        avg_accuracy = running_accuracy / log_interval
        
        print(f"Iter {iter_num}: loss {avg_loss:.4f}, accuracy {avg_accuracy:.2f}%, lr {lr:.6f}, time {dt*1000:.2f}ms")
        
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": avg_loss,
                "train/accuracy": avg_accuracy,
                "lr": lr,
            })
        
        # Reset running metrics
        running_loss = 0
        running_accuracy = 0
    
    # Increment iteration counters
    iter_num += 1
    local_iter_num += 1

# Clean up DDP if used
if ddp:
    destroy_process_group()

print(f"Training completed after {iter_num} iterations.")

# Final evaluation
if master_process:
    val_loss, val_acc = evaluate()
    print(f"Final validation: loss {val_loss:.4f}, accuracy {val_acc:.2f}%")