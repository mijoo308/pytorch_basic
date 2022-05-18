import numpy as np
import torch


''' [ view : 원소의 수를 유지하면서 텐서의 크기 변경 ] '''

t = np.array([[[0,1,2],
                [3,4,5]],
                [[6,7,8],
                [9,10,11]]])

ft = torch.FloatTensor(t)
print(ft.shape) # 2x2x3

''' (1) 3차원 텐서 ->  2차원 텐서로 변경 '''
print(ft.view([-1,3])) # (?,3) 의 크기로 변경하라는 의미 (?는 원소의 개수에 맞게 파이토치가 알아서 계산해 줌)
print(ft.view([-1,3]).shape)

''' (2) 3차원 텐서의 크기 변경 '''
# 텐서의 차원은 유지하되 shape을 바꾸는 작업
print(ft.view([-1,1,3])) # (?, 1, 3) 의 크기로 변경
print(ft.view([-1,1,3]).shape)



''' [ Squeeze : 1인 차원을 제거 ] '''
ft = torch.FloatTensor([[0],[1],[2]]) # 3x1
print(ft.squeeze()) # 3x1에서 두번째 차원이 1이므로 squeeze를 사용하면 (3,)
print(ft.squeeze().shape)


'''[ Unsqueeze : 특정 위치에 1인 차원을 추가 ]'''
ft = torch.Tensor([0,1,2]) # shape : (3)인 tensor
print(ft.unsqueeze(0))
print(ft.unsqueeze(0).shape) # (1,3)

print(ft.unsqueeze(1))
print(ft.unsqueeze(1).shape) # (3,1)


'''[ Type Casting ]'''
lt = torch.LongTensor([1,2,3,4])
print(lt)
print(lt.float()) # >> tensor([1., 2., 3., 4.])

bt = torch.ByteTensor([True,False,False,True])
print(bt)
print(bt.long())
print(bt.float())

''' [ Concatenate (cat) ] '''
# 딥러닝에서 주로 모델의 입력 또는 중간 연산에서 두 개의 텐서를 연결하는 경우에 쓰임
x = torch.FloatTensor([[1,2],[3,4]])
y = torch.FloatTensor([[5,6],[7,8]])

# mf []
print(torch.cat([x,y], dim=0)) # 0번째 차원을 늘려라 (4,2)
print(torch.cat([x,y], dim=1)) # 1번째 차원을 늘려라 (2,4)


''' [ Stacking (stack) ]'''
x = torch.FloatTensor([1,4])
y = torch.FloatTensor([2,5])
z = torch.FloatTensor([3,6])

# mf []
print(torch.stack([x,y,z])) # default (dim=0)
# tensor([[1., 4.],
#         [2., 5.],
#         [3., 6.]]) 


print(torch.stack([x,y,z], dim=1))
# tensor([[1., 2., 3.],
#         [4., 5., 6.]])


'''[ ones_like / zeors_like ]'''
x = torch.FloatTensor([[0,1,2],[2,1,0]])
print(torch.ones_like(x)) # 같은 shape의 1로 채워진 tensor
print(torch.zeros_like(x)) # 같은 shape의 0으로 채워진 tensor


'''[ 덮어쓰기 연산 ( _ ) ]'''
x = torch.FloatTensor([[1,2],[3,4]])
print(x.mul(2))
print(x)

print(x.mul_(2)) # 계산도 하고 그 결과도 저장
print(x)










