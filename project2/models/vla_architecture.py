"""
VLA Transformer Architecture
19M parameters, 6-layer encoder
"""
import torch
import torch.nn as nn

class FullFidelityVLA(nn.Module):
    def __init__(self, num_classes, input_dim=512, hidden_dim=512, num_layers=6, nhead=8):
        super().__init__()
        self.pos_enc = nn.Parameter(torch.randn(1, 30, input_dim) * 0.02)
        encoder = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, 
            dim_feedforward=2048, dropout=0.4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        x = x + self.pos_enc
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.head(x)
