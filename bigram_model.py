import torch
import torch.nn.functional as F

words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set("".join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0 

block_size = 3
X, Y = [], []
for w in words[:5]:
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]

X = torch.tensor(X)
Y = torch.tensor(Y)
num = X.nelement()
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27), requires_grad=True)
for epoch in range(20):
    xenc = F.one_hot(X, num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    loss = -probs[torch.arange(num), Y].log().mean() + 0.01*(W**2).mean()
    print(loss.item())
    W.grad = None
    loss.backward()
    W.data += -10 * W.grad