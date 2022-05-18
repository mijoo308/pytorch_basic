import numpy as np
import torch

''' 1D with pytorch'''
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t)
print(t.dim())  # dimmension
print(t.size()) # shape (=size)
print(t.shape) # shape

''' 2D with pytorch'''
t = torch.FloatTensor([[1.,2.,3.],
                        [4.,5.,6.,],
                        [7.,8.,9.],
                        [10.,11.,12.]])

print(t)
print(t.dim()) # 2
print(t.shape) # torch.size([4, 3])

''' Broadcasting '''
# vector + scalar
m1 = torch.FloatTensor([[1,2]])
m2 = torch.FloatTensor([3]) # -> [3, 3]
print(m1 + m2)

# 2x1 vector + 1x2 vector
m1 = torch.FloatTensor([[1,2]])
m2 = torch.FloatTensor([[3],[4]])
print(m1 + m2)



''' Matrix Multiplication (matmul)'''
m1 = torch.FloatTensor([[1,2],[3,4]]) # 2x2
m2 = torch.FloatTensor([[1],[2]]) # 2x1
print('m1 shape :', m1.shape)
print('m2 shape :', m2.shape)
print(m1.matmul(m2))  # 2x1

''' element-wise Multiplication (* or mul)'''
m1 = torch.FloatTensor([[1,2],[3,4]]) # 2x2
m2 = torch.FloatTensor([[1],[2]])  # 2x1
print(m1 * m2)
print(m1.mul(m2)) # 마찬가지로 broadcasting 됨


''' Mean '''
t = torch.FloatTensor([1,2])
print(t.mean())

t = torch.FloatTensor([[1,2],
                       [3,4]])
print(t.mean()) # (1+2+3+4) // 4

# dim=0은 첫번째 차원 (=행)
# dim=0 을 인자로 주면 해당 차원을 제거함
# (2,2) -> (1,2)
print(t.mean(dim=0)) # 1과 3의 평균을 구하고, 2와 4의 평균을 구함
# >> tensor([2., 3.])

''' Sum '''
t = torch.FloatTensor([[1,2],[3,4]])
print(t.sum())
print(t.sum(dim=0)) # 행 제거 : [4,6]
print(t.sum(dim=1)) # 열 제거 :  [3,7]
print(t.sum(dim=-1)) # 열 제거 : [3,7]

# Q. 결과는 무조건 [,,,,,]임?


''' max & argmax(최대값 index) '''
# max는 index 까지 같이 있음
t = torch.FloatTensor([[1,2],[3,4]])
print(t.max()) # (4)
print(t.max(dim=0)) # [3,4] , index ~
print(t.max(dim=1)) # [2,4] , index ~

print('max :', t.max(dim=0)[0]) # max
print('argmax : ', t.max(dim=0)[1]) # argmax