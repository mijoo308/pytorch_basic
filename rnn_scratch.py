import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

time_steps = 10  # 시점의 수. NLP에서 보통 문장 길이
input_size = 4  # 입력의 차원. NLP에서 보통 단어 벡터의 차원
hidden_size = 8 # 은닉 상태의 크기

inputs = np.random.random((time_steps, input_size)) # 2D 텐서
hidden_state_t = np.zeros((hidden_size,)) # 초기 은닉 상태는 0벡터로 초기화

print(hidden_state_t)

Wx = np.random.random((hidden_size, input_size)) # 8 x 4 2D 텐서
Wh = np.random.random((hidden_size, hidden_size)) # 8 x 8 2D 텐서
b = np.random.random((hidden_size,)) # (8,) 크기의 1D 텐서

total_hidden_states = []

for input_t in inputs:
    output_t = np.tanh(np.dot(Wx, input_t) + np.dot(Wh, hidden_state_t) + b)
    total_hidden_states.append(list(output_t)) # 각 시점의 은닉 상태 값 축적
    print(np.shape(total_hidden_states))
    hidden_state_t = output_t

total_hidden_states = np.stack(total_hidden_states, axis=0)
print(total_hidden_states)


''' pytorch '''

input_size = 5
hidden_size = 8

inputs = torch.Tensor(1, 10, 5)
cell = nn.RNN(input_size, hidden_size, batch_first=True)
# num_layers param으로 깊이 설정 가능
# bidirectional param으로 bidirectional rnn 구현 가능

output, _status = cell(inputs)

print(output.shape) # 모든 time step의 hidden state
print(_status.shape) # 최종 time step의 hidden state
