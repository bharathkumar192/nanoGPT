# Configuration for training VisionGPT on MNIST/FashionMNIST without model compilation

# Output directory
out_dir = 'out-vision-mnist'

# Wandb logging
wandb_log = False
wandb_project = 'vision-gpt'
wandb_run_name = 'vision-gpt-mnist'

# Data
batch_size = 64
gradient_accumulation_steps = 1

# Model configuration - smaller model for faster training
n_layer = 6
n_head = 8
n_embd = 256
dropout = 0.1
bias = True
img_size = 28
patch_size = 4
in_channels = 1
n_classes = 20  # 10 from MNIST + 10 from FashionMNIST

# Training parameters
max_iters = 2000
learning_rate = 1e-3
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 100
lr_decay_iters = 2000
min_lr = 1e-4

# Evaluation
eval_interval = 100
log_interval = 10

# System - disable compilation to avoid errors
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'float32'  # Use float32 for more stable training
compile = False    # Disable compilation to avoid errors
