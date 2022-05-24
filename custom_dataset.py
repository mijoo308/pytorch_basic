
''' custom dataset 기본 구조 

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self) -> None: # 데이터셋 전처리
        super().__init__()

    def __len__(self): # 데이터셋의 길이 (sample 수)

    def __getitem__(self): # 데이터셋에서 특정 1개의 샘플을 가져옴

'''

# custom dataset으로 linear regression 구현 ------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.x_data = [[73, 80, 75],
                    [93, 88, 93],
                    [89, 91, 90],
                    [96, 98, 100],
                    [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self,idx): # tensor 형태로 return
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x,y


dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
# dataloader를 선언하고 나면 dataset을 직접 사용하는 일은 없음

model = nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)


nb_epochs = 20
for epoch in range(nb_epochs+1):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples ###
        # H(x)
        pred = model(x_train)
        # cost
        cost = F.mse_loss(pred, y_train)
        # H(x) 갱신
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, batch_idx+1, len(dataloader), cost.item()
        ))

# test
new_var = torch.FloatTensor([[73, 80, 75]])
pred_y = model(new_var)
print('pred : ', pred_y.item())
