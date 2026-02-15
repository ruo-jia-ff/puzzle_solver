import torch
import torch.nn as nn
import torch.nn.functional as F


# Improved Edge Encoder
class EdgeEncoder(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, dim=1)

        return x


# Siamese Compatibility Network
class SiameseCompatibility(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.encoder = EdgeEncoder(embedding_dim)

    def forward(self, x1, x2):
        e1 = self.encoder(x1)
        e2 = self.encoder(x2)
        similarity = torch.sum(e1 * e2, dim=1)
        return similarity
