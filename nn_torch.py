import torch
import torch.nn as nn
import torch.optim as op

class SimpleNN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.dense1 = nn.Linear(in_features=in_features, out_features=128)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(in_features=128, out_features=out_features)

    def forward(self, x):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)

    



