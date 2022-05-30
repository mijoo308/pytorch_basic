import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

 # Activation Function - Nonlinear function
 
# sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y)
plt.plot([0,0], [1,0], ':')
plt.title('sigmoid function')
plt.show()
 # >>  gradient vanishing
 
# hyperbolic tangent (tanh)
x = np.arange(-5.0, 5.0, 0.1)
y = np.tanh(x)

plt.plot(x, y)
plt.plot([0,0], [1,-1], ':')
plt.axhline(y=0, color='orange', linestyle='--')
plt.title('Tanh Function')
plt.show()
# (0,0) 을 중심으로 해서 반환 값의 변화폭이 더 큼 -> sigmoid 보다 기울기 소실이 적음


# ReLU
def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)

plt.plot(x, y)
plt.plot([0,0], [5,0], ':')
plt.title('ReLU Function')
plt.show()

# Leaky ReLU
# 입력 값이 음수일 때도 0이 되지 않음
a = 0.1
def leaky_relu(x):
    return np.maximum(a*x, x)

x = np.arange(-5.0, 5.0, 0.1)
y = leaky_relu(x)

plt.plot(x, y)
plt.plot([0,0], [5,0], ':')
plt.title('Leaky ReLU Function')
plt.show() 


# softmax
x = np.arange(-5.0, 5.0, 0.1)
y = np.exp(x)/np.sum(np.exp(x))

plt.plot(x, y)
plt.title('Softmax Function')
plt.show()
