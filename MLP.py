import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layers_hidden, activation=nn.ReLU, output_activation=None, dropout=0.0):
        super().__init__()

        layers = []
        for i in range(len(layers_hidden) - 1):
            layers.append(nn.Linear(layers_hidden[i], layers_hidden[i + 1]))

            is_last = i == len(layers_hidden) - 2
            if not is_last:
                layers.append(activation())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            elif output_activation is not None:
                layers.append(output_activation())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
    def compute_loss(self, x: torch.Tensor, y: torch.Tensor, loss_fn = torch.nn.MSELoss()):
        pred = self.forward(x)

        mse_loss = loss_fn(pred, y)
        return mse_loss