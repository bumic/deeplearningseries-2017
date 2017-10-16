'''
	Author: Justin Chen
	Date: 9.25.17

	Workshop 3: 
	Neural Networks

	Boston University
	Machine Intelligence Community
'''

import os, torch
from datetime import datetime
from data import Dataset
from torch.autograd import Variable
import torch.optim as optim


class Trainer(object):
	def __init__(self, model, config):
		self.model   = model
		self.lr      = config['lr']
		self.epoches = config['epochs']
		self.batches = config['batchs']
		self.samples = config['samples']
		self.dataset = Dataset(self.samples, self.batches)


	def train(self):
		losses = []
		validate = []
		h = []
		t = []

		optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

		for e in range(self.epoches):
			for b, (sample, target) in enumerate(self.dataset.load_dataset('train')):

				sample, target = Variable(sample), Variable(target)

				# clear the local gradients
				optimizer.zero_grad()

				# compute hypothesis and calculate the loss
				hypo = self.model(sample)
				loss = self.model.loss(hypo, target)
				
				# backpropagate gradients of loss w.r.t. parameters
				loss.backward()

				# track loss
				losses.append(loss.data.tolist()[0])

				# update weights
				optimizer.step()

			validate.append(self.validate())

		torch.save(self.model.state_dict(), os.path.join('save', datetime.now().strftime('%m-%d-%Y-%H-%M') + '.pth'))

		self.dataset.subplots_2D(2, 'Training', y_data_list=[losses, validate], subplt_titles=['Cumulative Loss', 'Validation'],
					   x_label=['Iterations', 'Epochs'],
					   y_label=['Cumulative Loss', 'Accuracy'])
		self.dataset.show('Binary Spiral')


	def validate(self):
		self.model.eval()
		losses = []
		correct = 0
		cumulative_loss = 0

		validation_set = self.dataset.load_dataset('validation')

		for data, target in validation_set:
			data, target = Variable(data, volatile=True), Variable(target)
			output = self.model(data)
			cumulative_loss += self.model.loss(output, target, size_average=False).data[0]
			losses.append(cumulative_loss)

			# class of one-hot vector
			pred = output.data.max(1)[1]
			correct += pred.eq(target.data).sum()

		total = len(validation_set) * self.batches
		acc = 100. * correct / total

		return acc
