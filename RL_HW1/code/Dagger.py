import numpy as np
from abc import abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class MyData(Dataset):
	def __init__(self):
		self.data = []
		self.label = []
	def __len__(self):
		return len(self.data)
	def __getitem__(self, index):
		data = self.data[index]
		label = self.label[index]
		return data, label
	def append(self, data, label):
		if len(self.data) > 0:
			assert(data.shape == self.data[-1].shape)
		self.data.append(data)
		self.label.append(label)
		

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
		self.metric_name = "Accuracy"
		self.epochs = 10
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		
		self.model = NN(h, w, outputs).to(self.device)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
		self.criterion = nn.CrossEntropyLoss()
		self.metric_func = lambda y_pred, y_true: (y_pred.argmax(dim = 1) == y_true).float().mean()
		
	def predict(self, x):
		x = torch.tensor(x, dtype = torch.float)
		x = x.unsqueeze(0)
		x = x.permute(0, 3, 1, 2)
		x = x.to(self.device)
		self.model.eval()
		results = self.model(x).argmax(dim=1).item()
		return results
	
	def train_step(self, features, labels):
		self.model.train()
		self.optimizer.zero_grad()
		predictions = self.model(features)
		loss = self.criterion(predictions, labels)
		metric = self.metric_func(predictions, labels)
		loss.backward()
		self.optimizer.step()
		return loss.item(), metric.item()
	
	def valid_step(self, features, labels):
		self.model.eval()
		with torch.no_grad():
			predictions = self.model(features)
			loss = self.criterion(predictions, labels)
			metric = self.metric_func(predictions, labels)
		return loss.item(), metric.item()
	
	def train(self, data_set):
		dfhistory = pd.DataFrame(columns = ["epoch", "loss", self.metric_name, "val_loss", "val_" + self.metric_name]) 
		train_loader = DataLoader(data_set, batch_size = 64, shuffle = True, num_workers = 1)
		print("Start Training...")
		for epoch in range(self.epochs): 
        # 1，training loop-------------------------------------------------
			loss_sum = 0.0
			metric_sum = 0.0
			step = 1
			for step, (features, labels) in enumerate(train_loader, 1):
				features = features.to(self.device)
				labels = labels.to(self.device)
				loss, metric = self.train_step(features, labels)
				loss_sum += loss
				metric_sum += metric

        # 2，validation loop-------------------------------------------------
			val_loss_sum = 0.0
			val_metric_sum = 0.0
			val_step = 1
			for val_step, (features, labels) in enumerate(train_loader, 1):
				features = features.to(self.device)
				labels = labels.to(self.device)
				val_loss, val_metric = self.valid_step(features, labels)
				val_loss_sum += val_loss
				val_metric_sum += val_metric

        # 3，logging-------------------------------------------------
			info = (epoch, loss_sum / step, metric_sum / step, 
				val_loss_sum / val_step, val_metric_sum / val_step)
			dfhistory.loc[epoch] = info

			# print epoch-level logs
			print(("\nEPOCH = %d, loss = %.3f,"+ self.metric_name + \
			    "  = %.3f, valLoss = %.3f, "+"val"+ self.metric_name+" = %.3f") %info)

		print('Finished Training...')

		return dfhistory

class MyDaggerAgent(DaggerAgent):
	def __init__(self, necessary_parameters=None):
		super(DaggerAgent, self).__init__()
		# init your model
		self.model = Model(210, 160, 8)

	# train your model with labeled data
	def update(self, data_set):
		self.model.train(data_set)

	# select actions by your model
	def select_action(self, data_batch):
		label_predict = self.model.predict(data_batch)
		return label_predict


