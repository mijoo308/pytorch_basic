import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 현재 실습하고 있는 파이썬 코드를 재실행해도 다음에도 같은 결과가 나오도록 랜덤 시드를 줌
torch.manual_seed(1)

# 데이터
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

# 모델 초기화 (W와 b 초기화)
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.SGD([W,b], lr=0.01)

nb_epochs = 2000
for epoch in range(nb_epochs+1):

    # H(x) 계산  init : y = 0 * x + 0
    hypothesis = x_train * W + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad() # gradient를 0으로 초기화
    cost.backward() # cost function을 미분해 gradient 계산
    optimizer.step() # W와 b를 업데이트

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W:{:.3f}, b:{:.3f}, Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()))
