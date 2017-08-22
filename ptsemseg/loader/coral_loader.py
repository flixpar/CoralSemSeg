import cv2
import numpy as np

import os
import glob

from torch.utils import data

class CoralLoader(data.Dataset):

	n_classes = 9
	n_channels = 4
	MEAN_PIX = np.array([135.18068, 148.34402, 101.72406, 101.72406])

	def __init__(self, root_dir, split="training", img_size=800):

		self.root_dir = root_dir
		self.split = split
		self.img_size = img_size

		self.image_list = self.get_image_list()
		self.annotation_list = self.get_annotation_list()

		self.data = self.load_data()
		self.preprocess()
		self.downsample()

		self.add_channel()

	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return len(self.data)

	## Helpers:

	def get_image_list(self):
		image_fn_pattern = os.path.join(self.root_dir, self.split, "images", "*.png")
		filelist = glob.glob(image_fn_pattern)
		return sorted(filelist)

	def get_annotation_list(self):
		annotation_fn_pattern = os.path.join(self.root_dir, self.split, "annotations", "*.png")
		filelist = glob.glob(annotation_fn_pattern)
		return sorted(filelist)

	def load_data(self):
		images = []
		annotations = []

		# load images:
		for fn in self.image_list:
			img = cv2.imread(fn)
			images.append(img)

		# load annotations:
		for fn in self.annotation_list:
			img = cv2.imread(fn, 0)
			annotations.append(img)

		return list(zip(images, annotations))

	def preprocess(self):

		for img, _ in self.data:
			img = img.astype(np.float32)
			print(img.shape)
			img -= self.MEAN_PIX
			print(img.shape)
			img -= img.min(axis=2)
			print(img.shape)
			img /= img.max(axis=2)
			print(img.shape)

	def downsample(self):

		for img, mask in self.data:
			img = cv2.resize(img, (self.img_size, self.img_size), cv2.INTER_LANCZOS4)
			mask = cv2.resize(mask, (self.img_size, self.img_size), cv2.INTER_NEAREST)

	def add_channel(self):

		for img, _ in self.data:
			img = np.append(img, img[:,:,2], axis=2)
