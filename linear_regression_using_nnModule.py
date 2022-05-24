''' pytorch에서 제공하는 선형 회귀 모델 사용'''
''' 이전엔 H(x) hypothesis와  cost를 직접 작성했었음 '''

'''
import torch.nn as nn
model = nn.Linear(input_dim, output_dim)

import torch.nn.functional as F
cost = F.mse_loss(prediction, y_train)

'''

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# x_train = torch.FloatTensor([[1], [2], [3]])
# y_train = torch.FloatTensor([[2], [4], [6]])

# # 단순 선형 회귀이므로 input_dim=1, output_dim=1
# # model 선언
# model = nn.Linear(1,1)
# print(list(model.parameters()))

# # optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# nb_epochs = 2000
# for epoch in range(nb_epochs+1):
#     # H(x)
#     prediction = model(x_train) # model(x_train)
#     # cost
#     cost = F.mse_loss(prediction, y_train) # 구현되어 있는 mse_loss 사용

#     # H(x) 업데이트
#     optimizer.zero_grad() # 0으로 초기화
#     cost.backward() #미분
#     optimizer.step() # W, b값 업데이트

#     if epoch % 100 == 0:
#         print('Epoch{:4d}/{} Cost:{:.6f}'.format(
#             epoch, nb_epochs, cost.item()
#         ))


# # test
# new_var = torch.FloatTensor([[4.0]])
# pred_y = model(new_var) # forward 연산 (H(x)식에 입력 x로부터 예측된 y를 얻는 연산)
# print('학습 후 입력이 4일 때의 예측값: ', pred_y.item())

# print(list(model.parameters())) # W와 b 값 확인


#----------------다중 선형 회귀 -------------------------
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

model = nn.Linear(3,1) # input_dim=3, output_dim=1
print(list(model.parameters())) # 초기 W, b 값 확인

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

# optimizer
nb_epochs = 2000
for epoch in range(nb_epochs+1):
    # H(x)
    prediction = model(x_train) # = model.forward(x_train) 
    #cost
    cost = F.mse_loss(prediction, y_train)

    # H(x) 업데이트
    optimizer.zero_grad() # gradient 0으로 초기화 
    cost.backward() # 미분
    optimizer.step() # 업데이트

    print('Epoch {:4d}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, cost.item()
    ))

# test
new_var =  torch.FloatTensor([[73, 80, 75]])
pred_y = model(new_var) # forward
print("학습 후 입력이 73, 80, 75일 때의 예측값 :", pred_y.item())
print(list(model.parameters()))




