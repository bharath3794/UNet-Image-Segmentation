import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from model import UNet
from datagenerator import CustomDataGenerator


# Change this to your input images directory
images_dir = "./data/imgs v2/"

# Change this to your true masks directory 
labels_dir = "./data/masks v2/"

# we resized our images and masks to this size
resize = (256, 256) # (H, w)

# Categories considered. These are the same categories from cityscapes dataset
categories = [0, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

# number of channels to be predicted form the final layer of U-Net architecture
num_classes = len(categories)

def train_model(model, device, train_loader, valid_loader, optimizer, criterion, scheduler, epoch, softmax):
	'''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target 
    '''
	# Set model to train mode before each epoch
	model.train()

	# Empty list to store losses, accuracies, ious, dice_coeffs
	losses = []
	accuracies = []
	ious, dice_coeffs = [], []

	# Iterate over entire training samples (1 epoch)
	for batch_idx, batch_sample in enumerate(train_loader):
		data, target = batch_sample['image'], batch_sample['label']
		
		# Push data/label to correct device
		data, target = data.to(device, dtype=torch.float32), target.to(device, dtype=torch.long)

		# Do forward pass for current set of data
		predicted = model(data)

		# Compute loss based on criterion
		loss = criterion(predicted, target)

		print(f"Batch {batch_idx}: \n\ttrain_loss={loss}")

		# Store losses
		losses.append(loss.item())

		# Convert predicted probabilities to classes by applying softmax and argmax
		predicted = convert_predicted_to_class(predicted, softmax)
		
		# check how many of the predicted are correct
		correct = predicted.eq(target.view_as(predicted)).sum().item()
		
		# Current batch accuracy
		cur_acc = correct/(resize[0]*resize[1]*len(batch_sample['label']))
		
		# Store accuracies
		accuracies.append(cur_acc)
		
		# Compute the IoUs and Dice coeff. for each class of current batch
		iou, dice = compute_iou_dice(target, predicted, n_classes=num_classes)
		
		# store ious, dice coeff.
		ious.append(iou)
		dice_coeffs.append(dice)

		# find average ious and average dice coeff. for each batch
		iou_batch = np.around(np.nanmean(iou), decimals=4)
		dice_batch = np.around(np.nanmean(dice), decimals=4)

		print(f"\tacc={cur_acc}, IoU_batch={iou_batch}, dice_batch={dice_batch}\n")
		# Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
		optimizer.zero_grad()
		
		# Computes gradient based on final loss
		loss.backward()

		# Gradient Clipping
		nn.utils.clip_grad_value_(model.parameters(), 0.1)
		
		# Optimize model parameters based on learning rate and gradient 
		optimizer.step()
		
		# For each 256 * batch_size we will perform validation
		if batch_idx%256==0 and batch_idx!=0:
			# getting values from the validation 
			valid_loss, valid_acc, valid_iou, valid_dice, valid_avg_iou, valid_avg_dice = validation(model, device, valid_loader, softmax)
			print(f"\nVALIDATION: After {batch_idx+1} batches, \n\tvalid_loss={valid_loss}, valid_acc={valid_acc}, valid_avg_iou={valid_avg_iou}, valid_avg_dice={valid_avg_dice}")
			print(f"\tvalid_iou={valid_iou}")
			# to change the learning rate
			scheduler.step(valid_loss)
	iou_epoch, avg_iou = format_metrics(ious)
	dice_epoch, avg_dice = format_metrics(dice_coeffs)
	train_loss = np.around(np.mean(losses), decimals=4)
	train_acc = np.around(np.mean(accuracies), decimals=4)
	return train_loss, train_acc, iou_epoch, dice_epoch, avg_iou, avg_dice

def convert_predicted_to_class(pred, softmax):
	'''
	Input: predicted data from the model after each forward pass
	Output: resultant classes
	'''
	pred = pred.permute(0, 2, 3, 1)
	pred = softmax(pred)
	pred = torch.argmax(pred, dim=3)
	return pred

def validation(model, device, valid_loader, softmax):
	'''
    Validates the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    valid_loader: dataloader for validation samples.
    '''

    # Set model to eval mode to notify all layers.
	model.eval()
	valid_loss = []
	ious, dice_coeffs = [], []
	accuracies = []
	# Set torch.no_grad() to disable gradient computation and backpropagation
	with torch.no_grad():
		for batch_sample in valid_loader:
			data, target = batch_sample['image'], batch_sample['label']

			# Push data/label to correct device
			data, target = data.to(device, dtype=torch.float32), target.to(device, dtype=torch.long)
			
			# Predict the data by doing forward pass
			predicted = model(data)

			# Compute loss based on same criterion as training 
			cur_loss = F.cross_entropy(predicted, target).item()

			# Append loss to overall valid loss
			valid_loss.append(cur_loss)

			# Convert predicted probabilities to classes by applying softmax and argmax
			predicted = convert_predicted_to_class(predicted, softmax)

			# check how many of the predicted are correct
			correct = predicted.eq(target.view_as(predicted)).sum().item()

			# Current batch accuracy
			cur_acc = correct/(resize[0]*resize[1]*len(batch_sample['label']))

			# Store accuracies
			accuracies.append(cur_acc)

			# Compute the IoUs and Dice coeff. for each class of current batch
			iou, dice = compute_iou_dice(target, predicted, n_classes=num_classes)

			# store ious, dice coeff.
			ious.append(iou)
			dice_coeffs.append(dice)
	# set our model back to train mode
	model.train()
	avg_valid_loss = np.around(np.nanmean(valid_loss), decimals=4)
	val_acc = np.around(np.nanmean(accuracies), decimals=4)
	iou_epoch, avg_iou = format_metrics(ious)
	dice_epoch, avg_dice = format_metrics(dice_coeffs)
	return avg_valid_loss, val_acc, iou_epoch, dice_epoch, avg_iou, avg_dice


def compute_iou_dice(mask_true, mask_pred, n_classes=num_classes):
	'''
	Input: ture mask and predicted mask
	Output: ious and dice coeff for each class
	'''
	ious = [[] for i in range(n_classes)]
	dice = [[] for i in range(n_classes)]
	for cls in range(n_classes):
		pred_idxs = mask_pred == cls
		true_idxs = mask_true == cls
		intersection = [pred_idxs[i][true_idxs[i]].sum().item() for i in range(len(pred_idxs))]
		union = [pred_idxs[i].sum().item()+true_idxs[i].sum().item()-intersection[i] for i in range(len(pred_idxs))]
		dice_denom = np.array(union)+np.array(intersection)
		temp_iou = np.nanmean([intersection[i]/union[i] if union[i]!=0 else float('nan') for i in range(len(union))])
		ious[cls].append(temp_iou)
		temp_dice = np.nanmean([2*intersection[i]/dice_denom[i] if dice_denom[i]!=0 else float('nan') for i in range(len(dice_denom))])
		dice[cls].append(temp_dice)
	ious = [np.nanmean(ious[i]) for i in range(len(ious))]
	dice = [np.nanmean(dice[i]) for i in range(len(dice))]
	return ious, dice

def format_metrics(arr):
	cats = categories
	hashmap = {}
	avg = np.around(np.nanmean(arr, axis=0), decimals=4)
	for i in range(len(avg)):
		hashmap[cats[i]] = avg[i]
	return hashmap, np.around(np.nanmean(avg), decimals=4)


def run_main(config):
	# Check if cuda is available
	use_cuda = torch.cuda.is_available()

	# Set proper device based on cuda availability 
	device = torch.device("cuda" if use_cuda else "cpu")
	print("Torch device selected: ", device)

	# Initialize the model and send to device 
	model = UNet(n_channels=3, n_classes=num_classes)
	model.to(device)
	print(summary(model, (3, resize[0], resize[1])))
	# Initialize optimizer type
	optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate, weight_decay=1e-8, momentum=0.9)

	# Optionally, use a scheduler to change learning rate at certain interval manually
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
	# Initialize the criterion for loss computation 
	criterion = nn.CrossEntropyLoss(reduction='mean')
	
	softmax = nn.Softmax(dim=3)

	
	# If checkpoint is provided, load it to model
	if config.cp_dir:
		checkpoint = torch.load(config.cp_dir)
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])

	# Load datasets for training and testing
	dataset = CustomDataGenerator(config.image_dir, config.label_dir, categories[1:], resize)
	train_size = len(dataset) - config.valid_size
	train, valid = random_split(dataset, [train_size, config.valid_size])

	train_loader = DataLoader(train, batch_size = config.batch_size, 
	                            shuffle=True, num_workers=4, pin_memory=True)
	valid_loader = DataLoader(valid, batch_size = config.batch_size, 
	                            shuffle=False, num_workers=4, pin_memory=True)


	# Init variable to store best loss, can use for saving best model 
	best_loss = float('inf')

	# Create summary writer object in specified folder. 
	# Use same head folder and different sub_folder to easily compare between runs
	# Eg. SummaryWriter("my_logs/run1_Adam"), SummaryWriter("my_logs/run2_SGD")
	#     This allows tensorboard to easily compare between run1 and run2
	writer = SummaryWriter("runs/"+config.plot_file_name, comment=f'LR_{config.learning_rate}_BS_{config.batch_size}')

	# Run training for n_epochs specified in config 
	for epoch in range(1, config.num_epochs + 1):
		train_loss, train_acc, train_iou, train_dice, train_avg_iou, train_avg_dice = train_model(model, device, train_loader, valid_loader, optimizer, criterion, scheduler, epoch, softmax)
		valid_loss, valid_acc, valid_iou, valid_dice, valid_avg_iou, valid_avg_dice = validation(model, device, valid_loader, softmax)
		print(f"\nEPOCH: {epoch}")
		print(f"\ttrain_loss = {train_loss}, valid_loss = {valid_loss}")
		print(f"\ttrain_acc={train_acc}, valid_acc={valid_acc}")
		print(f"\ttrain_avg_iou = {train_avg_iou}, valid_avg_iou = {valid_avg_iou}")
		print(f"\ttrain_avg_dice = {train_avg_dice}, valid_avg_dice = {valid_avg_dice}")
		print(f"\ntrain_ious={train_iou}\nvalid_ious={valid_iou}")
		print(f"\ntrain_dices={train_dice}\nvalid_dices={valid_dice}")
		scheduler.step(valid_loss)

		writer.add_scalar('Loss/train', train_loss, epoch)
		writer.add_scalar('IoU/train', train_avg_iou, epoch)
		writer.add_scalar('Dice/train', train_avg_dice, epoch)
		writer.add_scalar('Loss/valid', valid_loss, epoch)
		writer.add_scalar('IoU/valid', valid_avg_iou, epoch)
		writer.add_scalar('Dice/valid', valid_avg_dice, epoch)
		writer.add_scalar('DiceLoss/train', 1-train_avg_dice, epoch)
		writer.add_scalar('DiceLoss/Valid', 1-valid_avg_dice, epoch)
		if valid_loss <= best_loss and config.save_dir:
			best_loss = valid_loss
			save_file_path = os.path.join(config.save_dir, 'model_{}_{:2.2f}.pth'.format(epoch, best_loss))
			states = {
			    'epoch': epoch,
			    'state_dict': model.state_dict(),
			    'optimizer': optimizer.state_dict(),
			    'best_loss': best_loss
			}

			try:
			    os.mkdir(config.save_dir)
			except:
			    pass

			torch.save(states, save_file_path)
			print('Model saved ', str(save_file_path))
	# Flush all log to writer and close 
	writer.flush()
	writer.close()
	print("Training finished")


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
	parser.add_argument('-lr', '--learning_rate',
	                    type=float, default=0.001,
	                    help='Initial learning rate.')
	parser.add_argument('-e', '--num_epochs',
	                    type=int,
	                    default=25,
	                    help='Number of epochs to run trainer.')
	parser.add_argument('-b', '--batch_size',
	                    type=int, default=16,
	                    help='Batch size. Must divide evenly into the dataset sizes.')
	parser.add_argument('-v', '--valid_size',
	                    type=int, default=2500,
	                    help='Validation Size')
	parser.add_argument('-s', '--save_dir',
	                    type=str,
	                    default='checkpoints',
	                    help='Directory to put runs for tensorboard visualization.')
	parser.add_argument('-p', '--plot_file_name',
	                    type=str,
	                    default='unet',
	                    help='Directory to put runs for tensorboard visualization.')
	parser.add_argument('-cp', '--cp_dir',
	                    type=str, default='',
	                    help='Directory of the saved checkpoint')

	config = None
	config, unparsed = parser.parse_known_args()

	print(f"Running UNet:\n\tLearning Rate={config.learning_rate}\n\tEpochs={config.num_epochs}\
	    \n\tBatch Size={config.batch_size}\n")

	run_main(config)