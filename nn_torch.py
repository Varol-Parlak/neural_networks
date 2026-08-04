import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#download the dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

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

model = SimpleNN(in_features=784, hidden_features=128, out_features=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 5
samples = len(train_dataset)

for epoch in range(EPOCHS):
    model.train()

    epoch_loss = 0
    epoch_acc = 0

    for x_batch, y_batch in train_loader:   
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        epoch_acc += (predictions == y_batch).float().mean().item()

    num_batches = len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {epoch_loss/num_batches:.4f} | Accuracy: {(epoch_acc/num_batches)*100:.2f}%")

def eval_model(model, test_loader):
    model.eval()

    test_acc=0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:

            outputs = model(x_batch)

            predictions = torch.argmax(outputs, dim=1)
            test_acc += (predictions == y_batch).float().mean().item()

    final_acc = test_acc / len(test_loader)
    print(f"Test Accuracy: {final_acc * 100:.2f}%")

eval_model(model, test_loader)
