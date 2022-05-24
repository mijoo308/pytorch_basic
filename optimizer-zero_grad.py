''' optimizer.zero_grad()가 필요한 이유 '''

# 미분을 통해 얻은 기울기를 이전 기울기 값에 누적시키는 경향이 있기 때문
import torch
w = torch.tensor(2.0, requires_grad=True) # requires_grad : 자동 미분 설정

nb_epochs = 20
for epoch in range(nb_epochs+1):
    z = 2*w
    z.backward() # 미분
    print('수식을 w로 미분한 값 : {}'.format(w.grad))
