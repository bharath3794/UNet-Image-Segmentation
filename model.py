import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
	def __init__(self, n_channels, n_classes, debug=False):
		super(UNet, self).__init__()
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.downsample = nn.MaxPool2d(kernel_size=2)
		self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		self.layer1_down = self.ConvLayer(self.n_channels, 64)
		self.layer2_down = self.ConvLayer(64, 128)
		self.layer3_down = self.ConvLayer(128, 256)
		self.layer4_down = self.ConvLayer(256, 512)
		self.layer5_down = self.ConvLayer(512, 512)
		self.layer6_up = self.ConvLayer(1024, 256, mid_channels=512)
		self.layer7_up = self.ConvLayer(512, 128, mid_channels=256)
		self.layer8_up = self.ConvLayer(256, 64, mid_channels=128)
		self.layer9_up = self.ConvLayer(128, 64)
		self.out = nn.Conv2d(64, self.n_classes, kernel_size=1)



	def ConvLayer(self, in_channels, out_channels, mid_channels=False):
		mid_channels = out_channels if not mid_channels else mid_channels
		conv = nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(mid_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)
		return conv
	def concat_features(self, up_feature, down_feature):
		assert up_feature.shape[2] == down_feature.shape[2]
		assert up_feature.shape[3] == down_feature.shape[3]
		concatted = torch.cat((down_feature, up_feature), dim=1)
		return concatted

	def forward(self, x):
		x1 = self.layer1_down(x)
		x2 = self.downsample(x1)
		x2 = self.layer2_down(x2)
		x3 = self.downsample(x2)
		x3 = self.layer3_down(x3)
		x4 = self.downsample(x3)
		x4 = self.layer4_down(x4)
		x5 = self.downsample(x4)
		x5 = self.layer5_down(x5)
		x = self.upsample(x5)
		x = self.concat_features(x, x4)
		x = self.layer6_up(x)
		x = self.upsample(x)
		x = self.concat_features(x, x3)
		x = self.layer7_up(x)
		x = self.upsample(x)
		x = self.concat_features(x, x2)
		x = self.layer8_up(x)
		x = self.upsample(x)
		x = self.concat_features(x, x1)
		x = self.layer9_up(x)
		output = self.out(x)
		return output