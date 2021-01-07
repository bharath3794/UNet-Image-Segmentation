import argparse
import os
import cv2
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torchsummary import summary
from model import UNet
from PIL import Image


# Set images directory
images_dir = "./data/game_imgs/input/"

# Set masks directory
labels_dir = "./data/game_imgs/mask/"

# we resized our images and masks to this size
resize = (256, 256) # (H, w)

# Categories considered. These are the same categories from cityscapes dataset
categories = [0, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
images_lst = sorted(glob.glob(images_dir+"*.png"))
labels_lst = sorted(glob.glob(labels_dir+"*.png"))
num_classes = len(categories)
mask_colors = {11: [70, 70, 70],  23: [70, 130, 180],  17: [153, 153, 153],  0: [0, 0, 0],  
21: [ 107, 142, 35],  15: [100, 100, 150],  5: [111, 74, 0],  
22: [152, 251, 152],  13: [190, 153, 153],  12: [102, 102, 156],  
24: [220, 20, 60],  6: [81, 0, 81],  27: [0, 0, 70],  
7: [128, 64, 128],  19: [250, 170, 30],  20: [220, 220, 0],  
4: [20, 20, 20],  26: [0, 0, 142],  32: [0, 0, 230],  
8: [244, 35, 232],  34: [0, 0, 142],  1: [0, 0, 0],  16: [150, 120, 90],  
14: [180, 165, 180],  28: [0, 60, 100],  31: [0, 80, 100],  25: [255, 0, 0],  
33: [ 119, 11, 32],  30: [0, 0, 110]}



def convert_predicted_to_class(pred, softmax):
	'''
	Input: predicted data from the model after each forward pass
	Output: resultant classes
	'''
	pred = pred.permute(0, 2, 3, 1)
	pred = softmax(pred)
	pred = torch.argmax(pred, dim=3)
	return pred

def save_to_image(img, orig, pred, n_image, file_dir):
	pred = pred[0]
	pred = pred.numpy()
	pred_img = np.zeros(shape=(pred.shape[0], pred.shape[1], 3), dtype='uint8')
	mask = np.zeros(shape=(orig.shape[0], orig.shape[1], 3), dtype='uint8')
	for cls in range(1, len(categories)):
		pred_img[pred==cls] = mask_colors[categories[cls]]
		mask[orig==cls] = mask_colors[categories[cls]]
	# h, w = pred.shape[:2]
	# pred_img = cv2.resize(pred_img, (2*w, 2*h), interpolation = cv2.INTER_AREA)
	# mask = cv2.resize(mask, (2*w, 2*h), interpolation = cv2.INTER_AREA)
	# Saving Images
	Image.fromarray(img).save(f"{file_dir}/{n_image}_input.png")
	Image.fromarray(mask).save(f"{file_dir}/{n_image}_gtmask.png")
	Image.fromarray(pred_img).save(f"{file_dir}/{n_image}_pred.png")



def run_main(config):
	# Check if cuda is available
	# use_cuda = torch.cuda.is_available()

	# Set proper device based on cuda availability 
	# device = torch.device("cuda" if use_cuda else "cpu")
	device = 'cpu'
	print("Torch device selected: ", device)

	# Initialize the model and send to device 
	model = UNet(n_channels=3, n_classes=num_classes)
	model.to(device)

	softmax = nn.Softmax(dim=3)

	# If checkpoint is provided, load it to model
	if config.cp_dir:
		checkpoint = torch.load(config.cp_dir)
		model.load_state_dict(checkpoint['state_dict'])

	try:
		os.mkdir(config.save_dir)
	except:
		pass

	# Set model to eval mode to notify all layers.
	model.eval()
	with torch.no_grad():
		for i in range(len(images_lst)):
			image = Image.open(images_lst[i])
			label = Image.open(labels_lst[i])
			image = image.resize(resize)
			label = label.resize(resize)
			image = np.array(image)
			image_copy = image.copy()
			label = np.array(label)
			image = image.transpose((2, 0, 1))
			if np.max(image) > 1:
				image = image/255
			image = np.array([image])
			image = torch.from_numpy(image).type(torch.FloatTensor)
			new_label = np.zeros(label.shape, dtype='uint8')
			for j in range(1, len(categories)):
				new_label[label==categories[j]] = j
			
			# Predict the data by doing forward pass
			predicted = model(image)

			# Convert predicted probabilities to classes by applying softmax and argmax
			predicted = convert_predicted_to_class(predicted, softmax)

			# Save predicted images 
			save_to_image(image_copy, new_label, predicted, i+1, config.save_dir)


if __name__ == '__main__':
	# Set parameters for Sparse Autoencoder
	parser = argparse.ArgumentParser('UNet')
	parser.add_argument('-i', '--image_dir',
	                    type=str,
	                    default=images_dir,
	                    help='Directory of images')
	parser.add_argument('-l', '--label_dir',
	                    type=str,
	                    default=labels_dir,
	                    help='Directory of labels')
	parser.add_argument('-s', '--save_dir',
	                    type=str, default='./predicted_masks/',
	                    help='Directory to save the image')
	parser.add_argument('-cp', '--cp_dir',
	                    type=str, default='',
	                    help='Directory of the saved checkpoint')

	config = None
	config, unparsed = parser.parse_known_args()

	print(f"Testing UNet:\n")

	run_main(config)