import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossTokenAttention(nn.Module):
    """
    Multi-head attention module for vision transformer
    """
    def __init__(self, embed_size=160, num_attention_heads=5, dropout_prob=0.1):
        super().__init__()
        self.embed_size = embed_size
        self.num_attention_heads = num_attention_heads
        self.dropout_prob = dropout_prob
        
        # Ensure embed_size is divisible by num_heads
        assert embed_size % num_attention_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        # Calculate dimension per head
        self.dimension_per_head = embed_size // num_attention_heads
        
        # Linear projections for query, key, value
        self.query_projection = nn.Linear(embed_size, embed_size)
        self.key_projection = nn.Linear(embed_size, embed_size)
        self.value_projection = nn.Linear(embed_size, embed_size)
        
        # Output projection
        self.output_projection = nn.Linear(embed_size, embed_size)
        
        # Dropout
        self.attention_dropout = nn.Dropout(dropout_prob)
        self.output_dropout = nn.Dropout(dropout_prob)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize projection layers with small standard deviation"""
        for projection in [self.query_projection, self.key_projection, 
                          self.value_projection, self.output_projection]:
            nn.init.normal_(projection.weight, mean=0.0, std=0.02)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)
    
    def transpose_for_attention(self, x):
        """
        Reshape tensor for multi-head attention
        
        Args:
            x: Tensor of shape [batch_size, seq_length, embed_size]
            
        Returns:
            Tensor of shape [batch_size, num_heads, seq_length, head_dim]
        """
        batch_size, seq_length, _ = x.shape
        
        # Reshape to separate heads
        x = x.view(batch_size, seq_length, self.num_attention_heads, self.dimension_per_head)
        
        # Transpose to [batch_size, num_heads, seq_length, head_dim]
        x = x.transpose(1, 2)
        
        return x
    
    def forward(self, x):
        """
        Apply multi-head attention on input tensor
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, embed_size]
            
        Returns:
            Output tensor of shape [batch_size, seq_length, embed_size]
        """
        batch_size, seq_length, _ = x.shape
        
        # Compute query, key, value
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)
        
        # Reshape for multi-head attention
        query = self.transpose_for_attention(query)
        key = self.transpose_for_attention(key)
        value = self.transpose_for_attention(value)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        
        # Scale attention scores
        attention_scores = attention_scores / math.sqrt(self.dimension_per_head)
        
        # Compute attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, value)
        
        # Reshape back to [batch_size, seq_length, embed_size]
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_length, self.embed_size)
        
        # Apply output projection
        output = self.output_projection(context)
        output = self.output_dropout(output)
        
        return output