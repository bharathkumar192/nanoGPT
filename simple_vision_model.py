"""
Simplified vision adaptation of the nanoGPT model for image classification tasks.
This version removes the RoPE implementation for simplicity.
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# Import the original GPT model components to extend them
from model import LayerNorm

class PatchEmbedding(nn.Module):
    """
    Image to Patch Embedding layer.
    Similar to ViT's approach, it divides an image into patches and embeds them.
    """
    def __init__(self, img_size=28, patch_size=4, in_channels=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Linear projection from patches to embedding dimension
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # CLS token for classification (similar to ViT)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Standard position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        """
        Forward pass of the patch embedding.
        Args:
            x: Input images of shape (B, C, H, W)
        Returns:
            Embedded patches of shape (B, n_patches + 1, embed_dim)
        """
        B = x.shape[0]
        
        # Extract patches and project them
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        return x

class MultiHeadAttention(nn.Module):
    """
    Simple multi-head attention for vision tasks (no causal mask).
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Key, query, value projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Dimensions
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        
        # Flash attention for efficiency
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dimensionality
        
        # Calculate query, key, values
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        
        # Compute attention
        if self.flash:
            # Use efficient flash attention (no causal mask for vision)
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False
            )
        else:
            # Manual implementation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
            
        # Reshape and project back
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        
        return y

class Block(nn.Module):
    """
    Transformer block for vision tasks.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class VisionGPTConfig:
    """
    Configuration class for VisionGPT.
    """
    img_size: int = 28  # Default for MNIST/FashionMNIST
    patch_size: int = 4
    in_channels: int = 1
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 256
    n_classes: int = 20  # 10 for MNIST + 10 for FashionMNIST
    block_size: int = None  # Will be calculated
    vocab_size: int = None  # Not used for vision
    dropout: float = 0.1
    bias: bool = True
    
    def __post_init__(self):
        # Calculate block_size based on image size and patch size
        self.block_size = (self.img_size // self.patch_size) ** 2 + 1  # +1 for cls token

class SimpleVisionGPT(nn.Module):
    """
    SimpleVisionGPT model: a simplified adaptation of nanoGPT for vision tasks.
    Uses patch embeddings and regular positional embeddings.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Patch embedding instead of token embedding
        self.patch_embed = PatchEmbedding(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.n_embd
        )
        
        # Dropout
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        # Final layer norm
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        
        # Classification head
        self.head = nn.Linear(config.n_embd, config.n_classes, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        # Report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    def get_num_params(self):
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
    
    def _init_weights(self, module):
        """Initialize weights similarly to the original GPT model."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, x, targets=None):
        """
        Forward pass of the model.
        Args:
            x: Input images of shape (B, C, H, W)
            targets: Optional target labels for computing the loss
        Returns:
            logits: Classification logits
            loss: Loss value if targets are provided, else None
        """
        # Get patch embeddings
        x = self.patch_embed(x)  # (B, n_patches + 1, n_embd)
        
        # Apply dropout
        x = self.drop(x)
        
        # Forward through all transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final layer norm
        x = self.ln_f(x)
        
        # Classification head (use only the CLS token)
        logits = self.head(x[:, 0])  # Only use the CLS token for classification
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure optimizers the same way as in the original GPT model.
        """
        # Start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # Filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # Create optim groups: weight tensors in matmuls + embeddings get weight decay, others don't
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # Use AdamW optimizer
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        
        return optimizer