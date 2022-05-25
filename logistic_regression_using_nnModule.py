import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# data
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

model = nn.Sequential(
    # H(x) = sigmoid(linear) = sigmoid( WX+b )
    nn.Linear(2,1),
    nn.Sigmoid()
)

optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    # H(x)
    hypothesis = model(x_train)
    # cost
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # H(x) 업데이트
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 10 == 0:
        # 0.5 기준으로 True/False로 변환
        pred = hypothesis>torch.FloatTensor([0.5]) # 0.5 고정?
        # 변환한 값이 정답이랑 같으면 True
        correct_pred = pred.float() == y_train

        # accuracy 계산
        acc = correct_pred.sum().item()/len(correct_pred)

        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(
            epoch, nb_epochs, cost.item(), acc * 100,
        ))


# 학습시킨 모델이 sigmoid 형태를 띠는지 확인
print(model(x_train))

# W와 b 확인
print(list(model.parameters()))
