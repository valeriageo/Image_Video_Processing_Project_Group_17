import torch
from torch import nn


class SimpleBaseline(nn.Module):
    """A minimal CNN baseline: 2 conv blocks and a small classifier.

    Parameters
    - num_classes: number of output classes
    - dropout: dropout probability used in the classifier
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    m = SimpleBaseline()
    print(m)
    import torch

    dummy = torch.randn(2, 1, 32, 32)
    print(m(dummy).shape)
