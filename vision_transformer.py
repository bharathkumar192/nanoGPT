import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageTransformer(nn.Module):
    """
    Vision Transformer for image classification
    """
    def __init__(
        self,
        img_resolution=28,
        patch_dimension=4,
        input_channels=1,
        num_classes=20,
        embed_size=160,
        num_layers=6,
        num_attention_heads=5,
        mlp_expansion_factor=4,
        dropout_prob=0.1
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.patch_dimension = patch_dimension
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.embed_size = embed_size
        self.num_layers = num_layers
        
        # Patch embedding
        self.patch_embedding = TokenizePatches(
            img_resolution=img_resolution,
            patch_dimension=patch_dimension,
            input_channels=input_channels,
            embed_size=embed_size
        )
        
        # Dropout for embeddings
        self.embedding_dropout = nn.Dropout(dropout_prob)
        
        # Transformer encoder blocks
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(
                embed_size=embed_size,
                num_attention_heads=num_attention_heads,
                mlp_expansion_factor=mlp_expansion_factor,
                dropout_prob=dropout_prob
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(embed_size)
        
        # Classification head
        self.classifier = nn.Linear(embed_size, num_classes)
        
        # Initialize classifier
        nn.init.normal_(self.classifier.weight, std=0.02)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)
        
        # Print model info
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Created Vision Transformer with {num_params:,} parameters")
    
    def forward(self, x, labels=None):
        """
        Forward pass for the vision transformer
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            labels: Optional target labels
            
        Returns:
            logits: Classification logits
            loss: Classification loss if labels are provided
        """
        # Get patch embeddings
        x = self.patch_embedding(x)
        
        # Apply dropout to embeddings
        x = self.embedding_dropout(x)
        
        # Apply transformer blocks
        for block in self.encoder_blocks:
            x = block(x)
        
        # Apply final normalization
        x = self.final_norm(x)
        
        # Extract class token representation
        class_token = x[:, 0]
        
        # Compute classification logits
        logits = self.classifier(class_token)
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        return logits, loss

# Import necessary modules from other files
from patch_embedding import TokenizePatches
from transformer_block import EncoderBlock