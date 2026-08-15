from micrograd_nn import MLP

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]# the dataset

ys = [1.0, -1.0, -1.0, 1.0] # the targets

n = MLP(3, [4, 4, 1]) # 3 inputs, 2 layers of 4 neurons, 1 output

epoch = 20
learning_rate = 0.05

for k in range(epoch): # the learning loop
    ypred = [n(x) for x in xs]
    
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    
    n.zero_grad()
    
    loss.backward()
    
    for p in n.parameters():
        p.data -= learning_rate * p.grad  
        
    print(f"Epoch {k:2d} | Loss: {loss.data:.4f}")

print("\nFinal Predictions:")
for ygt, yout in zip(ys, ypred):
    print(f"Target: {ygt:4.1f} | Predicted: {yout.data:.4f}")