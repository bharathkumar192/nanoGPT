import torch
import torch.nn as nn
import math

class TokenizePatches(nn.Module):
    """
    Convert images into patch tokens with a classification token
    """
    def __init__(self, img_resolution=28, patch_dimension=4, input_channels=1, embed_size=160):
        super().__init__()
        self.img_resolution = img_resolution
        self.patch_dimension = patch_dimension
        self.input_channels = input_channels
        self.embed_size = embed_size
        
        # Calculate patch count
        self.patch_count = (img_resolution // patch_dimension) ** 2
        
        # Create projection layer (convert patches to embeddings)
        self.projection = nn.Conv2d(
            in_channels=input_channels,
            out_channels=embed_size,
            kernel_size=patch_dimension,
            stride=patch_dimension
        )
        
        # Classification token
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_size))
        
        # Position embeddings
        self.position_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_count, embed_size)
        )
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize the module parameters"""
        # Initialize class token
        nn.init.normal_(self.class_token, std=0.02)
        
        # Initialize position embeddings
        nn.init.normal_(self.position_embed, std=0.02)
        
        # Initialize projection weights
        fan_in = self.projection.in_channels * self.projection.kernel_size[0] * self.projection.kernel_size[1]
        nn.init.normal_(self.projection.weight, mean=0.0, std=math.sqrt(2.0 / fan_in))
        
        # Initialize projection bias if present
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)
    
    def forward(self, x):
        """
        Convert images to patch embeddings
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Tensor of shape [batch_size, 1 + num_patches, embed_dim]
        """
        batch_size = x.shape[0]
        
        # Extract patch embeddings
        x = self.projection(x)  # Shape: [B, embed_size, H/patch_dim, W/patch_dim]
        x = x.flatten(2)        # Shape: [B, embed_size, num_patches]
        x = x.transpose(1, 2)   # Shape: [B, num_patches, embed_size]
        
        # Add classification token
        cls_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embeddings
        x = x + self.position_embed
        
        return x