import sys
sys.path.append('/Users/nhassen/opt/anaconda3/envs/env_battery/lib/python3.9/site-packages')
import torch

import gym

import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler

print('scikit-learn package version: {}'.format(sklearn.__version__))
# scikit-learn package version: 0.21.3

scaler = MinMaxScaler()
x_sample = [-90, 90]
scaler.fit(np.array(x_sample)[:, np.newaxis]) # reshape data to satisfy fit() method requirements
x_data = np.array([[66,74,89], [1,44,53], [85,86,33], [30,23,80]])

xx = np.random.normal(size = 100)
print(xx)
print(xx.reshape(-1,1))
print(scaler.transform(xx.reshape(-1,1)))

rnn = torch.nn.LSTM(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(input, (h0, c0))
print(output.shape)