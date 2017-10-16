'''
	Author: Justin Chen
	Date: 9.25.17

	Workshop 3: 
	Neural Networks

	Boston University
	Machine Intelligence Community
'''

from math import ceil
from numpy import pi, exp, cos, sin, concatenate, asarray
from numpy.random import normal, randn, choice, shuffle
from torch.autograd import Variable
from torch import FloatTensor, LongTensor
import matplotlib.pyplot as plt


class Dataset(object):
	def __init__(self, size, batch, train=0.7, val=0.1, test=0.2):
		self.dataset = None
		data = self.spiral_sample(n=size)
		t = int(train*size)
		v = int(val*size)
		self.size = size
		self.batch_size = int(batch)
		self.train = data[0:t]
		self.valid = data[t:t+v]
		self.test = data[t+v:]


	def to_batched(self, dataset):
		prev = 0
		batch_set = []
		total = len(dataset)

		# Must randomize dataset each epoch
		shuffle(dataset)

		for i in range(total/self.batch_size):
			points = dataset[prev:prev+self.batch_size] if prev+self.batch_size < total else dataset[prev:]
			x_coord = []
			y_coord = []
			
			for (x, y) in points:
				x_coord.append(x)
				y_coord.append(y)
			prev += self.batch_size

			batch_set.append([FloatTensor(x_coord), LongTensor(y_coord)])

		return batch_set


	def load_dataset(self, name):
		if name == 'train':
			return self.to_batched(self.train)
		elif name == 'validation':
			return self.to_batched(self.valid)
		elif name == 'test':
			return self.to_batched(self.test)
		else:
			raise Exception('data.load_dataset(): Invalid dataset')
		

	def spiral_sample(self, n=100):
		a = 0.5
		b = 0.6
		n = int(n/2)
		dataset = []

		# points
		pt = randn(n)

		# class 1
		x1 = a*exp(b*pt)*cos(pt+pi)
		y1 = a*exp(b*pt)*sin(pt+pi)

		# class 2
		x2 = a*exp(b*pt)*cos(pt)
		y2 = a*exp(b*pt)*sin(pt)

		# add some variance
		noise_x = normal(0, a*0.25, n)
		noise_y = normal(0, a*0.25, n)

		# class 1
		x1 = x1+noise_x
		y1 = y1+noise_y

		# one-hot vector
		label1 = 0 #[1, 0]
		dataset.extend([([x, y], label1) for (x, y) in zip(x1, y1)])

		# class 2
		x2 = x2+noise_y
		y2 = y2+noise_x

		label2 = 1 # [0, 1]
		dataset.extend([([x, y], label2) for (x, y) in zip(x2, y2)])
		
		shuffle(dataset)
		self.dataset = (x1, y1, x2, y2)
		
		return dataset

		
	def subplots_2D(self, num_plots, title, x_data_list=[], y_data_list=[], subplt_titles=[], x_label=[], y_label=[]):
		f, axarr = plt.subplots(num_plots, sharex=False)
		subplt_titles[0] = ' '.join([title, subplt_titles[0]])


		if len(x_data_list) == 0:
			for i, p in enumerate(axarr):
				p.set_title(subplt_titles[i])
				p.set_xlabel(x_label[i])
				p.set_ylabel(y_label[i])
				p.plot(y_data_list[i])
		else:
			for i, p in enumerate(axarr):
				p.set_title(subplt_titles[i])
				p.set_xlabel(x_label[i])
				p.set_ylabel(y_label[i])
				p.plot(x_data_list[i], y_data_list[i])

		plt.tight_layout()
		plt.show()


	def show(self, name='Dataset'):
		x1, y1, x2, y2 = self.dataset
		plt.figure(1)
		plt.title(name)
		plt.xlabel('x')
		plt.ylabel('y')
		plt.plot(x1, y1, 'bo', x2, y2, 'ro')
		plt.show()
