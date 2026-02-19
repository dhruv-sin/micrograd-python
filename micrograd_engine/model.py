import random
from engine_value import value
class neurone:
    def __init__(self,nin):
        self.w=[value(random.uniform(-1,1)) for _ in range(nin)]
        self.b= value(random.uniform(-1,1))
    def __call__(self,x):
        act=sum((wi*xi for wi,xi in zip(self.w,x)),self.b)
        out=act.tanh()
        return out 
    def parameter(self):
        return self.w + [self.b]

class layer:
    def __init__(self,nin,nout):
        self.neuron=[neurone(nin) for _ in range(nout)]
    def __call__(self,x):
        out=[neu(x) for neu in self.neuron]
        return out[0] if len(out) == 1 else out
    def parameter(self):
       return[p for neu in self.neuron for p in neu.parameter()]

class MLP:
    def __init__(self,nin,nout):
        size=[nin]+nout
        self.layer=[layer(size[i],size[i+1]) for i in range(len(nout))]
    def __call__(self, x):
        for layer in self.layer:
            x=layer(x)
        return x
    def parameter(self):
        return [p for lay in self.layer for p in lay.parameter()]

model = MLP(3, [4, 4, 1])

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0] 

# for i in range(5000):
#     ypred=[model(x) for x in xs ]
#     loss=sum((yt-yp)**2 for yt,yp in zip(ys,ypred))
#     for p in model.parameter():
#         p.grad=0
#     loss.backward()
#     for p in model.parameter():
#         p.data += -0.05 * p.grad
#     print(f"trial {i} loss-->{loss.data}")

# print(f"Predictions: {[y.data for y in ypred]}")

# loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
# print(f"Initial Loss: {loss.data}")
print("hi")