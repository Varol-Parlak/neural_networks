import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set("".join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0 

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

# num = X.nelement()
# g = torch.Generator().manual_seed(2147483647)
# W = torch.randn((27,27), requires_grad=True)
# for epoch in range(20):
#     xenc = F.one_hot(X, num_classes=27).float()
#     logits = xenc @ W
#     counts = logits.exp()
#     probs = counts / counts.sum(1, keepdim=True)
#     loss = -probs[torch.arange(num), Y].log().mean() + 0.01*(W**2).mean()
#     print(loss.item())
#     W.grad = None
#     loss.backward()
#     W.data += -10 * W.grad

g = torch.Generator().manual_seed(2147483647)
C = torch.rand((27, 2), generator=g)
W1 = torch.randn((6, 100), generator=g)
b1 = torch.rand(100, generator=g)
W2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]
epoch= 1000 
lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre

for p in parameters:
    p.requires_grad = True

lri, lossi = [], []

for i in range(epoch):
    ix = torch.randint(0, X.shape[0], (32,))

    emb = C[X[ix]]
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y[ix])
    print(loss.item())

    for p in parameters:
        p.grad = None 

    loss.backward()
    lr = lrs[i]
    for p in parameters:
        p.data += -lr * p.grad

    lri.append(lre[i])
    lossi.append(loss.item())

plt.plot(lri, lossi)
plt.show()