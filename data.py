import random

import numpy as np

from skimage import transform
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder
from torch.utils.data import random_split, DataLoader


class ImageFolderDataSet(ImageFolder):

	def __init__(self, root='./data/train', transforms=None, val_split=0.2, batch_size=16):
		super(ImageFolderDataSet, self).__init__(root=root)

		self.root = root
		self.val_split = val_split
		self.batch_size = batch_size

		if isinstance(transforms, (tuple, list)):
			train_transform = transforms[0]
			val_transform = transforms[1]
		else:
			train_transform = transforms
			val_transform = transforms
		
		val_length = int(len(self) * self.val_split)

		random.seed(42)
		val_indices = random.sample(range(len(self)), val_length)
		train_indices = [x for x in range(len(self)) if x not in val_indices]

		self.train_set = Subset(self, train_indices, train_transform)
		self.val_set = Subset(self, val_indices, val_transform)

		self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
		self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True)

	def __getitem__(self, index):
		item = super().__getitem__(index)
		data, label = item

		new_label = torch.zeros(len(self.classes), dtype=torch.float)
		new_label[label] = 1

		return (data, new_label)


class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        im, labels = self.dataset[self.indices[idx]]
        im = self.transform(im)
        return im, labels

    def __len__(self):
        return len(self.indices)


class Rescale(object):

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size

	def __call__(self, image):
		if type(image) is not np.ndarray:
			image = np.array(image)
		h, w = image.shape[:2]
		if isinstance(self.output_size, int):
			if h > w:
				new_h, new_w = self.output_size * h / w, self.output_size
			else:
				new_h, new_w = self.output_size, self.output_size * w / h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		image = transform.resize(image, (new_h, new_w))
		return image


class TestRescale(object):

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size

	def __call__(self, image):
		if type(image) is not np.ndarray:
			image = np.array(image)

		image = transform.resize(image, (int(self.output_size), int(self.output_size)))
		return image


class RandomCrop(object):

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size

	def __call__(self, image):
		h, w = image.shape[:2]
		new_h, new_w = self.output_size

		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)

		image = image[top: top + new_h, left: left + new_w]

		return image


class RandomFlip(object):

	def __init__(self, flip_prob=0.5):
		assert isinstance(flip_prob, (float, tuple))
		if isinstance(flip_prob, float):
			assert 0 <= flip_prob and 1 >= flip_prob
			self.flip_probs = (flip_prob, flip_prob)
		else:
			assert len(flip_prob) == 2
			for prob in flip_prob:
				assert 0 <= prob and 1 <= prob
			self.flip_probs = flip_prob

	def __call__(self, image):
		rand_probs = np.random.uniform(size=2)

		flip_dims = [1] if rand_probs[0] >= self.flip_probs[0] else []
		if rand_probs[1] >= self.flip_probs[1]:
			flip_dims.append(2)

		image = np.flip(image, axis=flip_dims).copy()

		return image


class RandomRotate(object):
	pass


class RandomNoise(object):

	def __init__(self):
		pass

	def __call__(self, image):
		image += np.random.normal(loc=0, scale=0.5, size=image.shape)
		return image

class ToTensor(object):

	def __call__(self, image):
		dims = image.shape
		if len(dims) == 3:
			image = image.transpose((2, 0, 1))
		else:
			image = image.transpose((0, 3, 1, 2))
		return torch.from_numpy(image).float()



