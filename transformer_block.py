import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    """
    Transformer encoder block for the vision transformer
    """
    def __init__(self, embed_size=160, num_attention_heads=5, 
                 mlp_expansion_factor=4, dropout_prob=0.1):
        super().__init__()
        self.embed_size = embed_size
        self.num_attention_heads = num_attention_heads
        self.mlp_expansion_factor = mlp_expansion_factor
        self.dropout_prob = dropout_prob
        
        # First layer normalization
        self.norm1 = nn.LayerNorm(embed_size)
        
        # Attention module
        self.attention = CrossTokenAttention(
            embed_size=embed_size,
            num_attention_heads=num_attention_heads,
            dropout_prob=dropout_prob
        )
        
        # Second layer normalization
        self.norm2 = nn.LayerNorm(embed_size)
        
        # MLP module
        self.mlp = FeedForwardNetwork(
            embed_size=embed_size,
            expansion_factor=mlp_expansion_factor,
            dropout_prob=dropout_prob
        )
    
    def forward(self, x):
        """
        Apply transformer block to input tensor
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, embed_size]
            
        Returns:
            Output tensor of shape [batch_size, seq_length, embed_size]
        """
        # First sub-layer: Multi-head attention with residual connection
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = residual + x
        
        # Second sub-layer: MLP with residual connection
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x

# Import necessary modules from other files
from attention_module import CrossTokenAttention
from mlp_module import FeedForwardNetwork