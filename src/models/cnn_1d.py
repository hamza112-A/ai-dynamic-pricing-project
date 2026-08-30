import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json


class Conv1DBlock(nn.Module):
    
    
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2  # Same padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class CNN1D(nn.Module):
    
    
    def __init__(
        self,
        n_features,
        n_classes,
        n_blocks=3,
        filters=64,
        kernel_size=3,
        dropout=0.3,
        pooling='max'
    ):
        
        super().__init__()
        
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_blocks = n_blocks
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.pooling = pooling
        
        # First conv block (input features -> filters)
        self.conv_blocks = nn.ModuleList([
            Conv1DBlock(1, filters, kernel_size, dropout)
        ])
        
        # Additional conv blocks
        for _ in range(n_blocks - 1):
            self.conv_blocks.append(
                Conv1DBlock(filters, filters, kernel_size, dropout)
            )
        
        # Global pooling
        self.pooling_type = pooling
        if pooling == 'both':
            pooled_features = filters * 2
        else:
            pooled_features = filters
        
        # Final classification layers
        self.fc = nn.Sequential(
            nn.Linear(pooled_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x):
        
        # Add channel dimension: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        
        # Apply conv blocks
        for block in self.conv_blocks:
            x = block(x)
        
        # Global pooling: (batch, filters, features) -> (batch, filters)
        if self.pooling_type == 'max':
            x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        elif self.pooling_type == 'avg':
            x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        elif self.pooling_type == 'both':
            max_pool = F.adaptive_max_pool1d(x, 1).squeeze(-1)
            avg_pool = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
            x = torch.cat([max_pool, avg_pool], dim=1)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
        
        # Classification
        x = self.fc(x)
        
        return x
    
    def get_num_parameters(self):
        
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNN1DModel:
    
    
    def __init__(
        self,
        n_features,
        n_classes,
        n_blocks=3,
        filters=64,
        kernel_size=3,
        dropout=0.3,
        pooling='max',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """Initialize model wrapper."""
        self.device = device
        self.model = CNN1D(
            n_features=n_features,
            n_classes=n_classes,
            n_blocks=n_blocks,
            filters=filters,
            kernel_size=kernel_size,
            dropout=dropout,
            pooling=pooling
        ).to(device)
        
        self.config = {
            'n_features': n_features,
            'n_classes': n_classes,
            'n_blocks': n_blocks,
            'filters': filters,
            'kernel_size': kernel_size,
            'dropout': dropout,
            'pooling': pooling
        }
        
        print(f"1D-CNN initialized on {device}")
        print(f"  Parameters: {self.model.get_num_parameters():,}")
    
    def predict(self, X):
        
        self.model.eval()
        
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        
        X = X.to(self.device)
        
        with torch.no_grad():
            logits = self.model(X)
            predictions = torch.argmax(logits, dim=1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X):
        
        self.model.eval()
        
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        
        X = X.to(self.device)
        
        with torch.no_grad():
            logits = self.model(X)
            probs = F.softmax(logits, dim=1)
        
        return probs.cpu().numpy()
    
    def save(self, path):
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        torch.save(self.model.state_dict(), path / 'model.pth')
        
        # Save configuration
        config_with_device = self.config.copy()
        config_with_device['device'] = str(self.device)
        
        with open(path / 'config.json', 'w') as f:
            json.dump(config_with_device, f, indent=2)
        
        print(f"Model saved to {path}")
    
    def load(self, path, device=None):
        
        path = Path(path)
        
        # Load configuration
        with open(path / 'config.json', 'r') as f:
            config = json.load(f)
        
        if device is not None:
            self.device = device
        elif 'device' in config:
            self.device = config['device']
        
        # Reinitialize model
        self.model = CNN1D(
            n_features=config['n_features'],
            n_classes=config['n_classes'],
            n_blocks=config['n_blocks'],
            filters=config['filters'],
            kernel_size=config['kernel_size'],
            dropout=config['dropout'],
            pooling=config['pooling']
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(
            torch.load(path / 'model.pth', map_location=self.device)
        )
        
        self.config = config
        
        print(f"Model loaded from {path}")
        return self


if __name__ == "__main__":
    # Quick test
    print("Testing 1D-CNN model...")
    
    # Create dummy data
    batch_size = 32
    n_features = 50
    n_classes = 5
    
    X = torch.randn(batch_size, n_features)
    y = torch.randint(0, n_classes, (batch_size,))
    
    # Initialize model
    model = CNN1DModel(
        n_features=n_features,
        n_classes=n_classes,
        n_blocks=3,
        filters=64,
        device='cpu'
    )
    
    # Forward pass
    predictions = model.predict(X.numpy())
    probabilities = model.predict_proba(X.numpy())
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Predictions: {predictions[:10]}")
    
    # Save and load
    model.save("test_cnn_model")
    
    model_loaded = CNN1DModel(n_features=n_features, n_classes=n_classes)
    model_loaded.load("test_cnn_model")
    
    print("\n✓ 1D-CNN model test passed")
