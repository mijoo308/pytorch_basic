import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.maual_seed_all(777)

x = torch.FloatTensor([[0,0], [0, 1], [1, 0], [1, 1]]).to(device)
y = torch.FloatTensor([[0],[1],[1],[0]]).to(device)

# 단층 퍼셉트론 구현
linear = nn.Linear(2, 1, bias=True)
sigmoid = nn.Sigmoid()
model = nn.Sequential(linear,sigmoid).to(device)

criterion = torch.nn.BCELoss().to(device) # 이진 분류에 사용했던 binary cross entropy 
optimizer = torch.optim.SGD(model.parameters(), lr=1)

for step in range(10001):
    optimizer.zero_grad()
    pred = model(x)

    cost = criterion(pred,y)
    cost.backward()
    optimizer.step()

    if step % 100 == 0:
        print(step, cost.item())

# >> 단층 퍼셉트론으로는 XOR 문제를 풀 수 없기 때문에 cost가 줄어들지 않음

# 학습 결과 확인
with torch.no_grad():
    pred = model(x)
    pred_to_print = (pred > 0.5).float()
    accuracy = (pred_to_print==y).float().mean()
    print('pred :', pred.detach().cpu().numpy())
    print('predicted :', pred_to_print.detach().cpu().numpy())
    print('실제값 : ', y.cpu().numpy())
    print('정확도 :', accuracy.item())
