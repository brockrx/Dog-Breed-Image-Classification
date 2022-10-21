import os
import pandas as pd

def main():
	data_path = './data/'
	train_path = data_path + 'train/'

	labels = pd.read_csv(data_path + 'labels.csv')

	breed_list = pd.unique(labels['breed'])
	for breed in breed_list:
		breed_path = train_path + breed + '/'
		os.mkdir(breed_path)

	for image_id, breed in zip(labels['id'], labels['breed']):
		current_file_name = train_path + image_id + '.jpg'
		new_file_name = train_path + breed + '/' + image_id + '.jpg'
		os.rename(current_file_name, new_file_name)

if __name__ == '__main__':
	main()


