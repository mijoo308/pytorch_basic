import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# # 데이터
# x_train = torch.FloatTensor([[1], [2], [3]])
# y_train = torch.FloatTensor([[2], [4], [6]])

# # class
# class LinearRegressionModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear = nn.Linear(1,1)

#     def forward(self, x):
#         return self.linear(x)


# model = LinearRegressionModel()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# nb_epochs = 2000
# for epoch in range(nb_epochs+1):
#     # H(x)
#     prediction = model.forward(x_train)

#     # cost
#     cost = F.mse_loss(prediction, y_train)

#     # H(x) 업데이트
#     optimizer.zero_grad()
#     cost.backward()
#     optimizer.step()

#     if epoch % 100 == 0:
#       print('Epoch {:4d}/{} Cost: {:.6f}'.format(
#           epoch, nb_epochs, cost.item()
#       ))

#-------------- 다중 선형 회귀 ---------------

# 데이터
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])


# class
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,1) #input_dim = 3, input_dim = 1

    def forward(self, x):
        return self.linear(x)


model = MultivariateLinearRegressionModel()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 2000
for epoch in range(nb_epochs+1):
    # H(x)
    pred = model(x_train)
    # cost
    cost = F.mse_loss(pred, y_train)
    # H(x) 갱신
    optimizer.zero_grad() # gradient 0 으로 초기화
    cost.backward() # 미분
    optimizer.step() # W값, b값 갱신

    if epoch % 100 == 0:
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))
