import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#download the dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

class SimpleNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return x

model = SimpleNN(in_features=784, hidden_features=128, out_features=10)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model().parameters, lr=0.001)

EPOCHS = 5
samples = len(train_dataset)

for epoch in range(EPOCHS):
    model.train()

    epoch_loss = 0
    epoch_acc = 0

    for x_batch, y_batch in train_loader:   
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        epoch_acc += (predictions == y_batch).float().mean().item()