import numpy as np
from abc import abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import pandas as pd

class DaggerAgent:
	def __init__(self,):
		pass

	@abstractmethod
	def select_action(self, ob):
		pass


# here is an example of creating your own Agent
class ExampleAgent(DaggerAgent):
	def __init__(self, necessary_parameters=None):
		super(DaggerAgent, self).__init__()
		# init your model
		self.model = None

	# train your model with labeled data
	def update(self, data_batch, label_batch):
		self.model.train(data_batch, label_batch)

	# select actions by your model
	def select_action(self, data_batch):
		label_predict = self.model.predict(data_batch)
		return label_predict

class NN(nn.Module):
	def __init__(self, h, w, outputs):
		super(NN, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, kernel_size = 5, stride = 2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size = 5, stride = 2)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 32, kernel_size = 5, stride = 2)
		self.bn3 = nn.BatchNorm2d(32)
		
		def conv2d_size_out(size, kernel_size = 5, stride = 2):
			return (size - (kernel_size - 1) - 1) // stride  + 1
		
		convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
		convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
		linear_input_size = convw * convh * 32
		self.head = nn.Linear(linear_input_size, outputs)

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		return self.head(x.view(x.size(0), -1))

class Model(object):
	def __init__(self, h, w, outputs):
		self.lr = 3e-5
		self.metricName = "Accuracy"
		self.epochs = 100
		self.model = NN(h, w, outputs)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
		self.criterion = nn.CrossEntropyLoss()
		self.metric_func = lambda y_pred, y_true: (y_pred.argmax(dim = 1) == y_true).float().mean()
	
	def predict(self, x):
		x = torch.tensor(x, dtype = torch.float)
		x = x.unsqueeze(0)
		x = x.permute(0, 3, 1, 2)
		self.model.eval()
		results = self.model(x).argmax(dim=1).item()
		return results
	
	def trainStep(self, features, labels):
		self.model.train()
		self.optimizer.zero_grad()
		predictions = self.model(features)
		loss = self.criterion(predictions, labels)
		metric = self.metric_func(predictions, labels)
		loss.backward()
		self.optimizer.step()
		return loss.item(), metric.item()
	
	def validStep(self, features, labels):
		self.model.eval()
		with torch.no_grad():
			predictions = self.model(features)
			loss = self.criterion(predictions, labels)
			metric = self.metric_func(predictions, labels)
		return loss.item(), metric.item()
	
	def train(self, features, labels):
		features = torch.tensor(features, dtype = torch.float)
		print(features.shape)
		features = features.permute(0, 3, 1, 2)
		labels = torch.tensor(labels, dtype = torch.long)
		
		dfhistory = pd.DataFrame(columns = ["epoch", "loss", self.metricName, "valLoss", "val" + self.metricName]) 
		print("Start Training...")
		for epoch in range(self.epochs): 
			loss, metric = self.trainStep(features, labels)
			valLoss, valMetric = self.validStep(features, labels)


			info = (epoch, loss, metric, valLoss, valMetric)
			dfhistory.loc[epoch] = info

			# print epoch-level logs
			print(("\nEPOCH = %d, loss = %.3f,"+ self.metricName + \
			    "  = %.3f, valLoss = %.3f, "+"val"+ self.metricName+" = %.3f") %info)

		print('Finished Training...')

		return dfhistory

class MyDaggerAgent(DaggerAgent):
	def __init__(self, necessary_parameters=None):
		super(DaggerAgent, self).__init__()
		# init your model
		self.model = Model(210, 160, 8)
		self.label2Action = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:11, 7:12}

	# train your model with labeled data
	def update(self, data_batch, label_batch):
		self.model.train(data_batch, label_batch)

	# select actions by your model
	def select_action(self, data_batch):
		label_predict = self.model.predict(data_batch)
		return self.label2Action(label_predict)


