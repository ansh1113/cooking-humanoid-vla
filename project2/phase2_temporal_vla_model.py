"""
PHASE 2: Temporal VLA Architecture
Enhanced transformer for action prediction
"""
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TemporalVLA(nn.Module):
    """
    Temporal Vision-Language-Action Model
    Input: Sequence of CLIP embeddings (T, D)
    Output: Action prediction
    """
    def __init__(
        self,
        embedding_dim=512,      # CLIP embedding dimension
        hidden_dim=512,
        num_actions=21,
        num_heads=8,
        num_layers=6,
        dropout=0.1
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        
        # Project CLIP embeddings to hidden dimension
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, embedding_dim) - CLIP embeddings
        Returns:
            logits: (batch_size, num_actions)
        """
        # Project to hidden dimension
        x = self.input_projection(x)  # (B, T, H)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer(x)  # (B, T, H)
        
        # Global average pooling over time
        x = x.mean(dim=1)  # (B, H)
        
        # Classification
        logits = self.classifier(x)  # (B, num_actions)
        
        return logits
    
    def predict_with_confidence(self, x):
        """Predict action with confidence score"""
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        confidence, predicted = torch.max(probs, dim=-1)
        return predicted, confidence

if __name__ == "__main__":
    # Test the model
    print("="*70)
    print("🧠 TEMPORAL VLA MODEL")
    print("="*70)
    
    model = TemporalVLA(
        embedding_dim=512,
        hidden_dim=512,
        num_actions=21,
        num_heads=8,
        num_layers=6
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Architecture:")
    print(f"   Embedding dim: 512")
    print(f"   Hidden dim: 512")
    print(f"   Num actions: 21")
    print(f"   Transformer layers: 6")
    print(f"   Attention heads: 8")
    
    print(f"\n📊 Parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 30
    x = torch.randn(batch_size, seq_len, 512)
    
    with torch.no_grad():
        logits = model(x)
        pred, conf = model.predict_with_confidence(x)
    
    print(f"\n✅ Test forward pass:")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {logits.shape}")
    print(f"   Predictions: {pred}")
    print(f"   Confidence: {conf}")
    
    print("\n" + "="*70)
    print("✅ MODEL READY!")
    print("="*70)
