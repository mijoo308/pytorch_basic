''' requters_grad = True '''

import torch

# 텐서에 대한 기울기를 저장하겠다는 의미
# w.grad에 w에 대한 미분값이 저장됨
w = torch.tensor(2.0, requires_grad=True)

y = w**2
z = 2*y + 5

z.backward()

print("수식을 w로 미분한 값 : {}".format(w.grad))
