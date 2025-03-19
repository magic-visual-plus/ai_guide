import torch.nn as nn

class ResMLP(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(ResMLP, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
            pass
        
        return x