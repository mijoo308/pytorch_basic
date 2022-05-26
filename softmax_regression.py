import torch
import torch.nn as n
import torch.nn.functional as F

# softmax 함수 확인해보기
z = torch.FloatTensor([1,2,3])
hypothesis = F.softmax(z, dim=0) # 첫번째 차원에 대해 (dim=0)
print(hypothesis) # 총 합이 1인 0~1 사이의 값을 가지는 벡터로 변환됨
print(hypothesis.sum())

# 3x5 크기의 임의의 텐서
z = torch.rand(3,5,requires_grad=True)
print(z)
hypothesis = F.softmax(z,dim=1) # 두번째 차원에 대해 (dim=1)
print(hypothesis[0].sum())
print(hypothesis[1].sum())
print(hypothesis[2].sum())


# one-hot 인코딩
y = torch.randint(5,(3,)).long()
print(y)

y_one_hot = torch.zeros_like(hypothesis)
# y.unsqueeze(1) : (3,) -> (3,1)
# >> tensor([[0],
            # [2],
            # [1]])
# _ : 덮어쓰기 연산이었음
y_one_hot.scatter_(1, y.unsqueeze(1), 1) # y.unsqueeze(1) 가 가리키는 곳에 1을 넣음
print(y_one_hot)


# cost function 구현 (= cross entropy)
cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
print(cost)

# softmax

# low-level
torch.log(F.softmax(z, dim=1))

# high-level ( softmax + log = F.log_softmax() )
F.log_softmax(z, dim=1)


# cost 함수
# low level
(y_one_hot * -torch.log(F.softmax(z, dim=1))).sum(dim=1).mean()
# high level 1
(y_one_hot * - F.log_softmax(z, dim=1)).sum(dim=1).mean()
# high level 2
F.nll_loss(F.log_softmax(z, dim=1), y)
# high level 3
F.cross_entropy(z, y)
