"""
Deep Neural Network mimarisi.
Esnek: katman sayısı, neuron, dropout, BN, activation configurable.
"""
import torch.nn as nn


ACTIVATIONS = {
    'ReLU': nn.ReLU,
    'SiLU': nn.SiLU,
    'Mish': nn.Mish,
    'GELU': nn.GELU,
}


class DNN(nn.Module):
    def __init__(
        self,
        input_dim=4,
        hidden_layers=None,
        dropout=0.0,
        use_batchnorm=False,
        activation='SiLU'
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [512, 64, 512, 512, 256]

        act_fn = ACTIVATIONS[activation]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x).squeeze(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
