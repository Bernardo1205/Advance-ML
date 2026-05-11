from torch import nn


class CirrhosisNN(nn.Module):
    def __init__(self, h1: int, h2: int, h3: int):
        super().__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(19, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, 3),
        )

    def forward(self, x):
        return self.linear_layers(x)