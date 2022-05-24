from random import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset # 텐서 데이터셋
from torch.utils.data import DataLoader # 데이터로더

''' 데이터 셋 -> 데이터 로더 '''

# 데이터
x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  90], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])


# 데이터셋
dataset = TensorDataset(x_train, y_train)
# 데이터셋 -> 데이터 로더
dataloader = DataLoader(dataset, batch_size=2, shuffle=True) # batch 여기서 설정


model = nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs+1):
    for batch_idx, samples in enumerate(dataloader): # batch 설정한 경우
        x_train, y_train = samples

        # H(x) 계산
        prediction = model(x_train)
        # cost 계산
        cost = F.mse_loss(prediction, y_train)
        # H(x) 업데이트
        optimizer.zero_grad() # gradient 0으로 초기화
        cost.backward() # 미분
        optimizer.step() # W,b 업데이트

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, batch_idx+1, len(dataloader),
        cost.item()
        ))
        # cost가 원래 batch 별로 들쭉날쭉함??
        # shuffle 순서에 따라서 성능이 달라질 수도 있나,,? 돌릴 때마다 결과가 다름


# test
new_var = torch.FloatTensor([[73, 80, 75]])
pred_y = model(new_var) # forward
print('학습 후 입력이 73, 80, 75일 때의 예측값 :', pred_y.item())
