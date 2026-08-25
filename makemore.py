import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set("".join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0 
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)
block_size = 3
def build_dataset(words):
    
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

Xtr, Ytr = build_dataset(words[:n1]) # The training percent of the words
Xdev, Ydev = build_dataset(words[n1:n2]) # The validation 
Xte, Yte = build_dataset(words[n2:]) # The test 
n_embed, n_hidden = 10, 200

g = torch.Generator().manual_seed(2147483647)
C = torch.rand((vocab_size, n_embed), generator=g) # Embedding table 27x10(27 chars 10 dimensions)

# Using He init
W1 = torch.randn((n_embed * block_size, n_hidden), generator=g) * ((5/3) / (30**0.5))
b1 = torch.randn(n_hidden, generator=g) * 0.01
W2 = torch.randn((n_hidden, vocab_size), generator=g) * ((5/3) / (300**0.5))
b2 = torch.randn(vocab_size, generator=g) * 0.01
# W3 = torch.randn((200,27), generator=g) * 0.01
# b3 = torch.randn(27, generator=g) * 0
# parameters = [C, W1, b1, W2, b2, W3, b3]
parameters = [C, W1, b1, W2, b2]
epoch, batchsize = 10000, 32
lossi, stepi = [], [] 

for p in parameters:
    p.requires_grad = True

for i in range(epoch):
    ix = torch.randint(0, Xtr.shape[0], (batchsize,))

    emb = C[Xtr[ix]] # embedding the chars into vectors
    embcat = emb.view(emb.shape[0], -1) # concat the vectors
    h1 = torch.tanh(embcat @ W1 + b1) # hidden layer activation
    # h2 = torch.tanh(h1 @ W2 + b2)
    logits = h1 @ W2 + b2 # output layer 
    loss = F.cross_entropy(logits, Ytr[ix]) # loss func
    
    for p in parameters:
        p.grad = None # zero_grad() so gradients dont accumulate

    loss.backward()

    lr = 0.1 if i < 30000 else 0.01 # decaying learning rate

    for p in parameters:
        p.data += -lr * p.grad

    lossi.append(loss.log10().item())
    stepi.append(i)


plt.plot(stepi,lossi)
plt.show()