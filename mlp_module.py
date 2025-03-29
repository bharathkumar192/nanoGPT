import torch
import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    """
    MLP module for transformer block
    """
    def __init__(self, embed_size=160, expansion_factor=4, dropout_prob=0.1):
        super().__init__()
        self.embed_size = embed_size
        self.expansion_factor = expansion_factor
        self.dropout_prob = dropout_prob
        
        # Calculate expanded dimension
        self.expanded_dim = embed_size * expansion_factor
        
        # MLP layers
        self.network = nn.Sequential(
            # First linear layer (expand dimension)
            nn.Linear(embed_size, self.expanded_dim),
            
            # Activation function
            nn.GELU(),
            
            # Dropout
            nn.Dropout(dropout_prob),
            
            # Second linear layer (contract dimension)
            nn.Linear(self.expanded_dim, embed_size),
            
            # Final dropout
            nn.Dropout(dropout_prob)
        )
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize the MLP layers"""
        # Get the linear layers
        linear_layers = [module for module in self.network if isinstance(module, nn.Linear)]
        
        # Initialize weights
        for layer in linear_layers:
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """
        Apply MLP to input tensor
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, embed_size]
            
        Returns:
            Output tensor of shape [batch_size, seq_length, embed_size]
        """
        return self.network(x)