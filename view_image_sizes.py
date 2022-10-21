import os
import pandas as pd
from skimage import io

def main():
	data_path = './data/'
	train_path = data_path + 'train/'

	labels = pd.read_csv(data_path + 'labels.csv')

	breed_list = pd.unique(labels['breed'])

	size_list = []
	for image_id, breed in zip(labels['id'], labels['breed']):
		file_name = train_path + breed + '/' + image_id + '.jpg'
		image = io.imread(file_name)
		size = image.shape
		if size not in size_list:
			print(size)
			size_list.append(size)

if __name__ == '__main__':
	main()


