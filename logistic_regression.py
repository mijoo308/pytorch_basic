'''
logistic Regression : 이진 분류에 사용

H(x) = Wx+b -> H(x) = sigmoid(Wb+b)

'''

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


''' sigmoid 함수 확인 '''
# def sigmoid(x):
#     return 1/(1+np.exp(-x))

# x = np.arange(-5.0, 5.0, 0.1)
# y = sigmoid(x)

# plt.plot(x,y,'g')
# plt.plot([0,0],[1,0], ':') # 가운데 점선 표시
# plt.title('Sigmoid Function')
# plt.show()

''' W값에 따른 경사도 변화 확인 '''
# x = np.arange(-5.0,5.0,0.1)
# y1 = sigmoid(0.5*x)
# y2 = sigmoid(x)
# y3 = sigmoid(2*x)

''' W값이 클수록 경사가 커짐 '''
# plt.plot(x,y1, 'r', linestyle='--')
# plt.plot(x,y2, 'g')
# plt.plot(x,y3, 'b', linestyle='--')
# plt.plot([0,0],[1,0], ':')
# plt.title('Sigmoid Function')
# plt.show()

''' b값의 변화에 따른 좌,우 이동 '''
# x = np.arange(-5.0, 5.0, 0.1)
# y1 = sigmoid(x+0.5)
# y2 = sigmoid(x)
# y3 = sigmoid(x+1.5)

# plt.plot(x,y1, 'r', linestyle='--')
# plt.plot(x,y2, 'g')
# plt.plot(x,y3, 'b', linestyle='--')
# plt.plot([0,0],[1,0], ':')
# plt.title('Sigmoid Function')
# plt.show()

''' ------------ logistic regrssion 구현 ----------------------'''
# data
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# 모델 초기화
W = torch.zeros((2,1), requires_grad=True)
b = torch.zeros(1, requires_grad = True)

# optimizer
optimizer = torch.optim.SGD([W,b], lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs+1):
    # H(x) = sigmoid(XW + b)
    hypothesis = torch.sigmoid(x_train.matmul(W) + b)

    # cost 함수
    # H(x) = Wx + b -> H(x) = sigmoid(Wx+b) 이므로 non-convex 그래프가 됨
    # 다른 cost 함수가 필요
    # 정답이 0일 때와 1일 때 모두를 다룰 수 있는 함수
    # "binary_cross_entropy"
    cost = -(y_train*torch.log(hypothesis)+(1-y_train)*torch.log(1-hypothesis)).mean()
    # ^ convex optimization 모양


    # H(x) 업데이트
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# test---------------------------------------------
test = torch.sigmoid(x_train.matmul(W) + b)
print(test)
# W,b 로는 sigmoid의 모양을 결정한 것

# pred = test>=torch.FloatTensor([0.5])
pred = test>=0.5 # 표현
print(pred)
