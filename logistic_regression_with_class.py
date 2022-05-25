import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# class 구현
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
# data
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)


model = BinaryClassifier()
optimizer = torch.optim.SGD(model.parameters(), lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs+1):
    # H(x)
    hypothesis = model(x_train)
    #cost
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # H(x) 업데이트
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch%10==0:
        pred  = hypothesis>0.5
        correct_pred = pred.float() == y_train

        # accuracy
        acc = correct_pred.sum()/len(correct_pred)

        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(
            epoch, nb_epochs, cost.item(), acc * 100,
        ))
