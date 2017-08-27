import cv2
import tifffile as tiff
import numpy as np
import scipy.misc as misc

import matplotlib.pyplot as plt

import os
import glob

import torch
from torch.utils import data

class CoralLoader(data.Dataset):

	n_classes = 9
	n_channels = 4
	MEAN_PIX = np.array([135.18068, 148.34402, 101.72406, 124.72406])

	def __init__(self, root_dir, split="training", img_size=800):

		self.root_dir = root_dir
		self.split = split
		self.img_size = img_size
		self.colors = colors = [np.random.rand(3) for i in range(self.n_classes)]

		self.image_list = self.get_image_list()
		self.annotation_list = self.get_annotation_list()

		print("{} images in loader.".format(self.__len__()))

	def __getitem__(self, index):
		img = cv2.imread(self.image_list[index])
		mask = cv2.imread(self.annotation_list[index], 0)

		img, mask = self.downsample(img, mask)

		plt.imshow(img)
		plt.show()

		img = self.addWhitenedChannel(img.copy())

		plt.imshow(img[:,:,3])
		plt.show()

		img = self.preprocess(img)

		print(img[:,:,0].min())
		print(img[:,:,0].min())
		print(img[:,:,1].min())
		print(img[:,:,1].min())
		print(img[:,:,2].max())
		print(img[:,:,2].max())
		print(img[:,:,3].max())
		print(img[:,:,3].max())

		img, mask = self.convert(img, mask)
		return img, mask

	def __len__(self):
		return len(self.image_list)

	## Helpers:

	def get_image_list(self):
		image_fn_pattern = os.path.join(self.root_dir, self.split, "images", "*.png")
		filelist = glob.glob(image_fn_pattern)
		return sorted(filelist)

	def get_annotation_list(self):
		annotation_fn_pattern = os.path.join(self.root_dir, self.split, "annotations", "*.png")
		filelist = glob.glob(annotation_fn_pattern)
		return sorted(filelist)

	def preprocess(self, img):
		img = img.astype(np.float32)
		img -= self.MEAN_PIX
		img -= img.min(axis=(0,1))
		img /= img.max(axis=(0,1))
		return img

	def downsample(self, img, mask):
		# img = cv2.resize(img, (self.img_size, self.img_size), cv2.INTER_LANCZOS4)
		# mask = cv2.resize(mask, (self.img_size, self.img_size), cv2.INTER_NEAREST)
		img = misc.imresize(img, (self.img_size, self.img_size), 'lanczos')
		mask = misc.imresize(mask, (self.img_size, self.img_size), 'nearest')
		return img, mask

	def convert(self, img, mask):
		img = img.transpose(2, 0, 1)
		img = torch.from_numpy(img).float()
		mask = torch.from_numpy(mask).long()
		return img, mask

	def decode_segmap(self, im):
		out = np.zeros((self.img_size, self.img_size, 3))
		for i in range(self.img_size):
			for j in range(self.img_size):
				out[i,j] = self.colors[im[i,j]]
		return out

	def addWhitenedChannel(self, img):
		# gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
		gray = np.average(img, weights=[0.114, 0.587, 0.299], axis=2)
		zca_matrix = zca_whitening_matrix(gray)
		zca_img = np.dot(zca_matrix, gray)
		zca_img = zca_img.reshape((zca_img.shape[0], zca_img.shape[1], 1))
		out = np.concatenate((img, zca_img), axis=2)
		return out

def zca_whitening_matrix(X):
	"""
	Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
	INPUT:  X: [M x N] matrix.
		Rows: Variables
		Columns: Observations
	OUTPUT: ZCAMatrix: [M x M] matrix
	"""
	# Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
	sigma = np.cov(X, rowvar=True) # [M x M]
	# Singular Value Decomposition. X = U * np.diag(S) * V
	U,S,V = np.linalg.svd(sigma)
		# U: [M x M] eigenvectors of sigma.
		# S: [M x 1] eigenvalues of sigma.
		# V: [M x M] transpose of U
	# Whitening constant: prevents division by zero
	epsilon = 1e-5
	# ZCA Whitening matrix: U * Lambda * U'
	ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]
	return ZCAMatrix
