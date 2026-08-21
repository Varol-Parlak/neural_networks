import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set("".join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0 

def build_dataset(words):

    block_size = 3
    X, Y = [], []

    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return X, Y

random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

g = torch.Generator().manual_seed(2147483647)
C = torch.rand((27, 10), generator=g)
W1 = torch.randn((30, 500), generator=g) * ((5/3) / (30**0.5))
b1 = torch.rand(500, generator=g) * 0.01
W2 = torch.randn((500, 27), generator=g) * 0.01
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]
epoch = 50000
lossi, stepi = [], []

for p in parameters:
    p.requires_grad = True

for i in range(epoch):
    ix = torch.randint(0, Xtr.shape[0], (32,))

    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])
    
    for p in parameters:
        p.grad = None 

    loss.backward()

    lr = 0.1 if i < 30000 else 0.01

    for p in parameters:
        p.data += -lr * p.grad

    lossi.append(loss.log10().item())
    stepi.append(i)

emb = C[Xdev]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)
print(loss)

emb = C[Xte]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Yte)
print(loss)

plt.plot(stepi,lossi)
plt.show()