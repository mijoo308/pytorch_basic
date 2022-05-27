import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

class SoftmaxClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4,3)  # output_dim = 3 -> class 개수

    def forward(self,x):
        return self.linear(x)


# data
x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

# one-hot encoding
y_one_hot = torch.zeros(8,3)
y_one_hot.scatter_(1, y_train.unsqueeze(1), 1) # one-hot encoding
print(y_one_hot.shape)


# model
model = SoftmaxClassifier()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    # H(x)
    pred = model(x_train)
    # cost
    cost = F.cross_entropy(pred, y_train)

    # H(x) 업데이트
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
