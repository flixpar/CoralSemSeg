import cv2
import tifffile as tiff
import numpy as np

import os
import glob

import torch
from torch.utils import data

class MangroveLoader(data.Dataset):

	n_classes = 4
	n_channels = 4
	MEAN_PIX = np.array([100.51086747, 87.29517102, 105.75206194, 102.56490222])

	def __init__(self, root_dir, split="training", img_size=800):

		self.root_dir = root_dir
		self.split = split
		self.img_size = img_size
		self.colors = colors = [np.random.rand(3) for i in range(self.n_classes)]

		self.image_list = self.get_image_list()
		self.annotation_list = self.get_annotation_list()

		print("{} images in loader.".format(len(self.image_list)))

		self.data = self.load_data()
		self.preprocess()
		self.downsample()
		self.convert()

	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return len(self.data)

	## Helpers:

	def get_image_list(self):
		image_fn_pattern = os.path.join(self.root_dir, self.split, "images", "*.tif")
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
			img = tiff.imread(fn)
			images.append(img)

		# load annotations:
		for fn in self.annotation_list:
			img = cv2.imread(fn, 0)
			annotations.append(img)

		return list(zip(images, annotations))

	def preprocess(self):
		self.data = [(self.preprocess_img(img), mask) for img, mask in self.data]

	def preprocess_img(self, img):
		img = img.astype(np.float32)
		img -= self.MEAN_PIX
		img -= img.min(axis=(0,1))
		img /= img.max(axis=(0,1))
		return img

	def downsample(self):
		images = [cv2.resize(img, (self.img_size, self.img_size), cv2.INTER_LANCZOS4) for img,_ in self.data]
		masks = [cv2.resize(mask, (self.img_size, self.img_size), cv2.INTER_NEAREST) for _,mask in self.data]
		self.data = zip(images, masks)

	def convert(self):
		self.data = [(img.transpose(2,0,1), mask) for img, mask in self.data]
		self.data = [(torch.from_numpy(img).float(), torch.from_numpy(mask).long()) for img, mask in self.data]

	def decode_segmap(self, im):
		out = np.zeros((self.img_size, self.img_size, 3))
		for i in range(self.img_size):
			for j in range(self.img_size):
				out[i,j] = self.colors[im[i,j]]
		return out
