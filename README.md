# Micrograd-Python: A Deep Learning Engine from First Principles

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 🚀 Overview
A tiny, pure-Python scalar-valued autograd engine and neural network library. This project implements backpropagation from scratch, providing a transparent look at how gradients flow through a computational graph to minimize loss.

**Why this exists:** As a 2nd-year CSE Honors student, I built this to move beyond high-level APIs like PyTorch/TensorFlow and master the underlying calculus and data structures that power modern AI.

## 🧠 Core Architecture
The engine tracks operations in a Directed Acyclic Graph (DAG). Each `Value` object stores its data and its gradient relative to the final output (typically the Loss).

* **Autograd Engine:** Manual implementation of the Chain Rule across addition, multiplication, and non-linear activation functions (tanh).
* **Neural Hierarchy:** Built custom `Neuron`, `Layer`, and `MLP` classes from the ground up to model biological neural structures.
* **Optimization:** Stochastic Gradient Descent (SGD) implementation for weight updates through manual backpropagation.

### The Math Behind the Code
The engine calculates the derivative of the Loss ($L$) with respect to any weight ($w$) using the chain rule:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w}$$

## 📊 Performance Results
The engine was tested on a non-linear regression task. By manually tuning the learning rate and architecture, I achieved the following convergence:

* **Initial Loss:** 7.5
* **Final Loss:** 9.06e-05
* **Training Depth:** Convergence achieved after 5,000 trials.
* **Optimization:** Verified successful gradient descent math through manual weight updates.

## 📂 Repository Structure
* **`micrograd_engine/`**: Core package containing `model.py` (Neural Network logic) and `engine_value.py` (Autograd logic).
* **`requirements.txt`**: Environment dependencies for reproducibility.
* **`.gitignore`**: Configured to exclude virtual environments and cache for repository hygiene.
* **`LICENSE`**: MIT License.

## 🛠️ Usage
```python
from micrograd_engine.model import MLP

# Define a 3-layer MLP: 3 inputs, two hidden layers of 4, 1 output
model = MLP(3, [4, 4, 1])

# Forward pass
inputs = [2.0, 3.0, -1.0]
out = model(inputs)

# Backward pass (The Magic)
out.backward()
