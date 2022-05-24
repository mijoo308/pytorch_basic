'''
x1,x2, ... : multivariable linear regression
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# torch.manual_seed(1)
# x1_train = torch.FloatTensor([[73],[93],[89],[96],[73]])
# x2_train = torch.FloatTensor([[80],[88],[91],[98],[66]])
# x3_train = torch.FloatTensor([[75],[93],[90],[100],[70]])
# y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

# w1 = torch.zeros(1, requires_grad=True)
# w2 = torch.zeros(1, requires_grad=True)
# w3 = torch.zeros(1, requires_grad=True)
# b = torch.zeros(1, requires_grad=True)

# optimizer = optim.SGD([w1,w2,w3,b], lr=1e-5)
# nb_epochs = 1000
# for epoch in range(nb_epochs):

#     # H(x) 계산
#     hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b

#     # cost 계산
#     cost = torch.mean((hypothesis-y_train)**2)

#     # cost로 H(x) 개선
#     optimizer.zero_grad()
#     cost.backward()
#     optimizer.step()

#     if epoch%100 == 0:
#         print('Epoch {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:.3f} Cost: {:.6f}'.format(
#             epoch, nb_epochs, w1.item(), w2.item(), w3.item(), b.item(), cost.item()
#         ))



# 행렬 연산으로 구현하기 -----------------------------------------------------------------------

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 80],
                             [96, 98, 100],
                             [73, 66, 70]])

y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

# print(x_train.shape) # 5x3
# print(y_train.shape) # 5x1

# H = XW + b
W = torch.zeros((3,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer
optimizer = optim.SGD([W, b], lr=1e-5)


nb_epochs = 20
for epoch in range(nb_epochs+1):
    # H(x)
    hypothesis = x_train.matmul(W) + b

    #cost
    cost = torch.mean((hypothesis - y_train)**2)

    # H 업데이트
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} hypothsis: {} Cost: {:.6f}'.format(
        epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item() # squeeze, detach
    ))
