import torch
from torch import nn
import numpy as np

class Trainer():

	def __init__(self, model, alpha, weight_decay, batch_size=16):
		self.model = model
		self.output_size = self.model.output_size
		self.alpha = alpha
		self.weight_decay = weight_decay
		self.batch_size = batch_size
		self.train_loss = []
		self.val_loss = []
		self.train_acc = []
		self.val_acc = []

		self.initialize_model()


	def initialize_model(self):
		self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.alpha, weight_decay=self.weight_decay)


	def train(self, epochs, train_dataloader, val_dataloader, device):
		self.model = self.model.to(device)
		for epoch in range(1,epochs+1):
			print("-----------------------------------")
			print("Epoch %d" % (epoch))
			print("-----------------------------------")
			running_loss = 0
			running_accuracy = 0
			train_count = 0
			for i, (inputs, labels) in enumerate(train_dataloader):
				self.optimizer.zero_grad()
				m = labels.size()[0]
				inputs, labels = inputs.to(device), labels.float().to(device)
				if len(inputs.size()) == 5:
					a, b, c, d, e = inputs.size()
					inputs = torch.reshape(inputs, (a*b, c, d, e))
				outputs = self.model(inputs)
				loss = self.loss_fn(outputs, labels)
				loss.backward()
				self.optimizer.step()
				running_accuracy += compute_accuracy(outputs, labels, reduction='sum')
				running_loss += loss.item()*m
				train_count += labels.size()[0]
			train_loss = running_loss/train_count
			train_accuracy =running_accuracy/train_count
			self.train_loss.append(train_loss)
			self.train_acc.append(train_accuracy)
			with torch.no_grad():
				running_loss = 0 
				running_accuracy = 0
				test_count = 0
				for i, (inputs, labels) in enumerate(val_dataloader):
					inputs, labels = inputs.to(device), labels.float().to(device)
					if len(inputs.size()) == 5:
						a, b, c, d, e = inputs.size()
						inputs = torch.reshape(inputs, (a*b, c, d, e))
					m = labels.size()[0]
					val_outputs = self.model(inputs)
					loss = self.loss_fn(val_outputs, labels)
					running_accuracy += compute_accuracy(val_outputs, labels, reduction='sum')
					running_loss += loss.item()*m
					test_count += labels.size()[0]
				val_loss = running_loss/test_count
				val_accuracy = running_accuracy/test_count
			self.val_loss.append(val_loss)
			self.val_acc.append(val_accuracy)
			np.set_printoptions(precision=2)
			print("Training: loss = %.4f | accuracy = %.2f" % (train_loss, train_accuracy))
			print("Validate: loss = %.4f | accuracy = %.2f" % (val_loss, val_accuracy))


def compute_accuracy(pred, target, reduction='mean'):
	assert reduction == 'mean' or reduction == 'sum'
	target = target.cpu().detach().numpy()
	pred = torch.argmax(pred, dim=1).cpu().detach().numpy()
	m, n = target.shape
	pred_class = np.zeros(target.shape)
	pred_class[np.arange(m), pred] = 1
	accuracy = np.sum(pred_class * target)
	if reduction == 'mean':
		accuracy /= m
	return accuracy*100


def loss_fn(pred, target):
	pass
