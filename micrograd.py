class Value():
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + self.other, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * self.other, (self, other))

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward

        return out

    def backward(self):

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.append(v)
                for children in v._prev:
                    build_topo(children)
                topo.append(v)

        self.grad = 1
        for v in reversed(topo):
            v._backward()


    def __neg__(self): 
        return self * -1

    def __radd__(self, other): 
        return self + other

    def __sub__(self, other): 
        return self + (-other)

    def __rsub__(self, other): 
        return other + (-self)

    def __rmul__(self, other): 
        return self * other

    def __truediv__(self, other): 
        return self * other**-1

    def __rtruediv__(self, other): 
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
