import math

import numpy as np

import torch
from torch import nn

class AlexNet(nn.Module):
	def __init__(self, input_size=(3,256,256), output_size=120):
		super(AlexNet, self).__init__()

		self.input_size = input_size
		self.output_size = output_size

		layer_list = [
			('conv', 96, 11, 4, 0),
			('relu'),
			('pool', 3, 2, 0),
			('conv', 256, 5, 1, 2),
			('relu'),
			('pool', 3, 2, 0),
			('conv', 384, 3, 1, 1),
			('relu'),
			('conv', 384, 3, 1, 1),
			('relu'),
			('conv', 256, 3, 1, 1),
			('relu'),
			('pool', 3, 1, 0),
			('drop', 0.5),
			('linear', 4096),
			('relu'),
			('drop', 0.5),
			('linear', 4096),
			('relu'),
			('linear', 120)
		]

		self.module_list = nn.ModuleList()

		for layer in layer_list:
			layer_type = layer[0]

			if layer_type == 'conv':
				layer_type, out_c, k, s, p = layer
				c, h, w = input_size
				self.module_list.append(nn.Conv2d(c, out_c, k, s, p))
				input_size = compute_conv_output_size(input_size, out_c, k, s, p)

			elif layer_type == 'pool':
				layer_type, k, s, p = layer
				c, h, w = input_size
				self.module_list.append(nn.MaxPool2d(k, stride=s, padding=p))
				input_size = compute_conv_output_size(input_size, c, k, s, p)

			elif layer_type == 'drop':
				layer_type, p = layer
				self.module_list.append(nn.Dropout(p=0.5))

			elif layer_type == 'relu':
				self.module_list.append(nn.ReLU())

			elif layer_type == 'linear':
				if isinstance(input_size, tuple):
					input_size = input_size[0] * input_size[1] * input_size[2]
				layer_size = layer[1]
				self.module_list.append(nn.Linear(input_size, layer_size))
				input_size = layer_size

			self.module_list.append(nn.Softmax(dim=1))


	def forward(self, x):
		if type(x) == np.ndarray:
			x = torch.from_numpy(x)
		out = x.float()

		if len(out.size()) == 2:
			out = torch.unsqueeze(out, 0)

		for module in self.module_list:
			if type(module) == nn.Linear and len(out.size()) == 4:
				out = torch.reshape(out, (out.size()[0], -1))
			out = module(out)

		return out


class VanillaCNN(nn.Module):
	def __init__(self, input_size=(3,256,256), output_size=120):
		super(VanillaCNN, self).__init__()

		self.input_size = input_size
		self.output_size = output_size

		layer_list = [
			('conv', 96, 11, 4, 0),
			('relu'),
			('pool', 3, 2, 0),
			('norm2d'),
			('conv', 256, 5, 1, 2),
			('relu'),
			('pool', 3, 2, 0),
			('norm2d'),
			('conv', 384, 3, 1, 1),
			('relu'),
			('norm2d'),
			('conv', 384, 3, 1, 1),
			('relu'),
			('norm2d'),
			('conv', 256, 3, 1, 1),
			('relu'),
			('pool', 3, 1, 0),
			('norm2d'),
			('drop', 0.5),
			('linear', 2048),
			('relu'),
			('norm'),
			('drop', 0.5),
			('linear', 2048),
			('relu'),
			('norm'),
			('linear', 120)
		]

		self.module_list = nn.ModuleList()

		for layer in layer_list:
			layer_type = layer[0]

			if layer_type == 'conv':
				layer_type, out_c, k, s, p = layer
				c, h, w = input_size
				self.module_list.append(nn.Conv2d(c, out_c, k, s, p))
				input_size = compute_conv_output_size(input_size, out_c, k, s, p)

			elif layer_type == 'pool':
				layer_type, k, s, p = layer
				c, h, w = input_size
				self.module_list.append(nn.MaxPool2d(k, stride=s, padding=p))
				input_size = compute_conv_output_size(input_size, c, k, s, p)

			elif layer_type == 'drop':
				layer_type, p = layer
				self.module_list.append(nn.Dropout(p=0.5))

			elif layer_type == 'relu':
				self.module_list.append(nn.ReLU())

			elif layer_type == 'linear':
				if isinstance(input_size, tuple):
					input_size = input_size[0] * input_size[1] * input_size[2]
				layer_size = layer[1]
				self.module_list.append(nn.Linear(input_size, layer_size))
				input_size = layer_size

			elif layer_type == 'norm2d':
				channels = input_size[1]
				self.module_list.append(nn.BatchNorm2d(channels))

			elif layer_type == 'norm':
				self.module_list.append(nn.BatchNorm1d(input_size))


	def forward(self, x):
		if type(x) == np.ndarray:
			x = torch.from_numpy(x)
		out = x.float()

		if len(out.size()) == 2:
			out = torch.unsqueeze(out, 0)

		for module in self.module_list:
			if type(module) == nn.Linear and len(out.size()) == 4:
				out = torch.reshape(out, (out.size()[0], -1))
			out = module(out)

		return out


class ResNetCNN(nn.Module):
	def __init__(self, input_size=(3,256,256), output_size=120):
		super(VanillaCNN, self).__init__()

		self.input_size = input_size
		self.output_size = output_size

		layer_list = [
			('conv', 128, 11, 4, 0),
			('relu'),
			('pool', 3, 2, 0),
			('norm2d'),
			('conv', 256, 5, 1, 2),
			('relu'),
			('pool', 3, 2, 0),
			('norm2d'),
			('conv', 256, 3, 1, 1),
			('relu'),
			('norm2d'),
			('conv', 128, 3, 1, 1),
			('relu'),
			('norm2d'),
			('conv', 64, 3, 1, 1),
			('relu'),
			('pool', 3, 1, 0),
			('norm2d'),
			('drop', 0),
			('linear', 2048),
			('relu'),
			('norm'),
			('drop', 0),
			('linear', 2048),
			('relu'),
			('norm'),
			('linear', 120)
		]

		self.module_list = nn.ModuleList()

		for layer in layer_list:
			layer_type = layer[0]

			if layer_type == 'conv':
				layer_type, out_c, k, s, p = layer
				c, h, w = input_size
				self.module_list.append(nn.Conv2d(c, out_c, k, s, p))
				input_size = compute_conv_output_size(input_size, out_c, k, s, p)

			elif layer_type == 'pool':
				layer_type, k, s, p = layer
				c, h, w = input_size
				self.module_list.append(nn.MaxPool2d(k, stride=s, padding=p))
				input_size = compute_conv_output_size(input_size, c, k, s, p)

			elif layer_type == 'drop':
				layer_type, p = layer
				self.module_list.append(nn.Dropout(p=0.5))

			elif layer_type == 'relu':
				self.module_list.append(nn.ReLU())

			elif layer_type == 'linear':
				if isinstance(input_size, tuple):
					input_size = input_size[0] * input_size[1] * input_size[2]
				layer_size = layer[1]
				self.module_list.append(nn.Linear(input_size, layer_size))
				input_size = layer_size

			elif layer_type == 'norm2d':
				channels = input_size[1]
				self.module_list.append(nn.BatchNorm2d(channels))

			elif layer_type == 'norm':
				self.module_list.append(nn.BatchNorm1d(input_size))


	def forward(self, x):
		if type(x) == np.ndarray:
			x = torch.from_numpy(x)
		out = x.float()

		if len(out.size()) == 2:
			out = torch.unsqueeze(out, 0)

		for module in self.module_list:
			if type(module) == nn.Linear and len(out.size()) == 4:
				out = torch.reshape(out, (out.size()[0], -1))
			out = module(out)

		return out

def compute_conv_output_size(input_size, out_channels, kernel_size, stride, padding):
	c, h, w = input_size
	
	if isinstance(kernel_size, (tuple, list)):
		k_h, k_w = kernel_size[0], kernel_size[1]
	else:
		k_h, k_w = kernel_size, kernel_size

	if isinstance(stride, (tuple, list)):
		s_h, s_w = stride[0], stride[1]
	else:
		s_h, s_w = stride, stride

	if isinstance(padding, (tuple, list)):
		p_h, p_w = padding[0], padding[1]
	else:
		p_h, p_w = padding, padding

	h = int(math.floor((h - k_h + 2 * p_h) / s_h) + 1)
	w = int(math.floor((w - k_w + 2 * p_w) / s_w) + 1)

	return (out_channels, h, w)