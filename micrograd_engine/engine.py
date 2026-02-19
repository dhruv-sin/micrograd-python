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

class Neuron:
    def __init__(self,nin):
        self.weight=[value(random.uniform(-1,1)) for _ in range(nin)]
        self.bias=value(random.uniform(-1,1))
    def __call__(self,x):
       act = sum((wi * xi for wi, xi in zip(self.weight, x)),self.bias)
       out = act.tanh()
       return out
    def parameters(self):
        return self.weight + [self.bias]
class Layers:
   def __init__(self,nin,nout):  
      self.neuron=[Neuron(nin)for _ in range(nout)]
   def __call__(self,x):
      out=[n(x) for n in self.neuron ]
      return out
   def parameters(self):
        return [p for n in self.neuron for p in n.parameters()]
class MLP:
   def __init__(self,nin,nouts):
      sz=[nin]+nouts
      self.layers=[Layers(sz[i],sz[i+1]) for i in range(len(nouts))]
   def __call__(self,x):
      for layer in self.layers:
         x=layer(x)
      return x
   def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
xs=[[-2,3,5],
   [4,5,-6],
   [-1,3,-8],
   [3,5,10]]
n=MLP(3,[4,4,1])
ypred=[n(x) for x in xs]
ytrue=[1,-1,1,-1]
loss= [(ygt-yout[0])**2 for ygt,yout in zip(ytrue,ypred)]

# # Training Loop
# for k in range(20):
#     # 1. Forward pass
#     ypred = [n(x) for x in xs]
#     # We sum the squared errors to get a single 'value' object
#     total_loss = sum((ygt - yout[0])**2 for ygt, yout in zip(ytrue, ypred))
    
#     # 2. Backward pass (The "Brain" updates its understanding)
#     # MUST zero out gradients first because they accumulate in our implementation
#     for p in n.parameters():
#         p.grad = 0.0
#     total_loss.backward()
    
#     # 3. Update (The Nudge)
#     learning_rate = 0.1
#     for p in n.parameters():
#         p.data += -learning_rate * p.grad
    
#     print(f"Step {k} | Loss: {total_loss.data:.4f}")
# Initialize a small brain
# 2 Inputs -> Layer 1 (3 Neurons) -> Layer 2 (Output)
# brain = MLP(2, [3, 1]) 

# print("--- STARTING MRI SCAN ---")

# # Level 1: The Whole Brain
# print(f"1. The Brain (MLP) has {len(brain.layers)} Layers.")

# # Level 2: The Layers
# for i, layer in enumerate(brain.layers):
#     print(f"\n  Layer {i}:")
#     print(f"  -> It contains {len(layer.neuron)} Neurons.")
    
#     # Level 3: The Neurons
#     for j, neuron in enumerate(layer.neuron):
#         print(f"    -> Neuron {j} has {len(neuron.weight)} Weights and 1 Bias.")
        
#         # Level 4: The Weights (The actual numbers)
#         # Let's peek at the first weight
#         first_weight = neuron.weight[0]
#         print(f"       [Weight 0 Data]: {first_weight.data:.4f}")
#         print(f"       [Weight 0 Grad]: {first_weight.grad:.4f}")

# print("\n--- SCAN COMPLETE ---")
