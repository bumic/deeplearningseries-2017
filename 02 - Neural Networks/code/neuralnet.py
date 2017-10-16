'''
	Author: Justin Chen
	Date: 9.25.17

	Workshop 3: 
	Neural Networks

	Boston University
	Machine Intelligence Community
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConnected(nn.Module):
	def __init__(self):
		super(FullyConnected, self).__init__()
		self.fc1 = nn.Linear(2, 40)
		self.fc2 = nn.Linear(40, 2)
		self.loss = F.nll_loss 

	def forward(self, x):
		return F.log_softmax(self.fc2(F.relu(self.fc1(x))))
