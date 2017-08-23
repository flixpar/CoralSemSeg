import sys
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import scipy.misc as misc
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import scores

class Namespace:
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)

args = Namespace(
	img_size = 768,
	batch_size = 1,
	dataset = "mangrove",
	model_path = "training/mangrove_coralnet_1_500.pkl",
	out_dir = "output/",
)

def test():

	# Setup Dataloader
	data_loader = get_loader(args.dataset)
	data_path = get_data_path(args.dataset)
	loader = data_loader(data_path, img_size=args.img_size)

	n_classes = loader.n_classes
	n_channels = loader.n_channels

	valloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=4, shuffle=True)

	# Setup Model
	model = torch.load(args.model_path)
	model.eval()

	if torch.cuda.is_available():
		model.cuda(0)

	for i, (images, labels) in enumerate(tqdm(valloader)):
		if torch.cuda.is_available():
			images = Variable(images.cuda(0))
			labels = Variable(labels.cuda(0))
		else:
			images = Variable(images)
			labels = Variable(labels)

		outputs = model(images)
		pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=1)
		gt = labels.data.cpu().numpy()

		for gt_, pred_ in zip(gt, pred):
			gt_path = args.out_dir + "gt{}.png".format(i)
			pred_path = args.out_dir + "pred{}.png".format(i)
			decoded_gt = loader.decode_segmap(gt_)
			decoded_pred = loader.decode_segmap(pred_)
			misc.imsave(gt_path, decoded_gt)
			misc.imsave(pred_path, decoded_pred)

if __name__ == '__main__':
	test()
