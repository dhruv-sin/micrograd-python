import math
import numpy as np
import matplotlib.pyplot as plt
import random
class value :
    def __init__(self,data,children=(),op=' '):
        self.data=data
        self.grad=0
        self._backward=lambda:None
        self._prev=set(children)
        self.op=op
    def __repr__(self):
        return f"value(data={self.data})"
    def __add__(self,other):
        other = other if isinstance(other, value) else value(other)
        out=value(self.data+other.data,(self,other),'+')
        def _backward():
            self.grad+=out.grad
            other.grad+=out.grad
        out._backward=_backward
        return out
    def __mul__(self,other):
        other = other if isinstance(other, value) else value(other)
        out=value(self.data*other.data,(self,other),'*')
        def _backward():
            self.grad+=other.data *out.grad
            other.grad+=self.data *out.grad
        out._backward=_backward
        return out
    def exp(self):
        x=self.data
        out=value((math.exp(x)),(self,),op=f"**{self.data}")
        def _backward():
            self.grad+=out.data*out.grad
        out._backward=_backward
        return out
    def __pow__(self, other):
      assert isinstance(other, (int, float)), "only supporting int/float powers for now"
      out = value(self.data**other, (self,), f"**{other}")
      def _backward():
           self.grad += (other * (self.data**(other - 1))) * out.grad
      out._backward = _backward
      return out
    
    def tanh(self):
        x = self.data
        # Calculate the forward pass
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = value(t, (self,), 'tanh')

        def _backward():
        # Using the derivative: (1 - tanh^2) * out.grad
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    def __neg__(self):
     return self * -1

    def __sub__(self, other): 
     return self + (-other)

    def __truediv__(self, other):
     return self * (other**-1)

    def __rtruediv__(self, other): 
     return other * (self**-1)
    def __rsub__(self, other): # handles: number - value
        return value(other) + (-self)

    def __radd__(self, other): # handles: number + value
        return self + other

    def __rmul__(self, other): # handles: number * value
        return self * other


    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()
    def __rmul__(self,other):return self*other
    def __radd__(self,other):return self+other