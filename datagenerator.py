import torch
import numpy as np
import os
import glob
from PIL import Image

class CustomDataGenerator(torch.utils.data.Dataset):
	def __init__(self, images_dir, labels_dir, categories, scale):
		self.images_dir = images_dir
		self.labels_dir = labels_dir
		self.scale = scale
		self.categories = categories
		self.images_lst = sorted(glob.glob(self.images_dir+"*.png"))
		self.labels_lst = sorted(glob.glob(self.labels_dir+"*.png"))

	def __len__(self):
		return len(self.images_lst)

	def __getitem__(self, i):
		image = Image.open(self.images_lst[i])
		label = Image.open(self.labels_lst[i])
		image = image.resize(self.scale)
		label = label.resize(self.scale)
		image = np.array(image)
		label = np.array(label)
		# Convert HWC to CHW
		image = image.transpose((2, 0, 1))
		if np.max(image) > 1:
			image = image/255
		new_label = np.zeros(label.shape, dtype='uint8')
		for i in range(len(self.categories)):
			new_label[label==self.categories[i]] = i+1
		dataset = {'image': torch.from_numpy(image).type(torch.FloatTensor), 
				   'label': torch.from_numpy(new_label).type(torch.IntTensor)}
		return dataset


