import random 
from micrograd_engine import Value

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters():
        return []

class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.weight = [Value(random.uniform(-1, 1) for _ in range(nin))]
        self.bias = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((xi*wi for wi, xi in zip(self.weight, x)), self.bias)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.weight + [self.bias]        

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.weight)})"

class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

