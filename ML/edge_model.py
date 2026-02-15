import torch.nn as nn
import torchvision.models as models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

"""
    Input: Concatenated edge strips (6-channel or 10-channel depending on preprocessing)
    Output: Single logit score indicating whether two edges belong together.
    Architecture - ResNet-18 backbone (modified first layer) & Fully connected classification head
"""

class EdgeNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights='DEFAULT')
        base.conv1 = nn.Conv2d(10, 64, 7, 2, 3, bias=False)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        f = self.backbone(x).squeeze()
        if f.dim() == 1:
            f = f.unsqueeze(0)
        return self.fc(f)
