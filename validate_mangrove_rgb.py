import sys
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
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
	img_size = 1024,
	batch_size = 1,
	dataset = "mangrove_rgb",
	model_path = "training/mangrove_linknet_2_500.pkl",
	max_samples = 100,
)

def validate():

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

	gts, preds = [], []
	for i, (images, labels) in tqdm(enumerate(valloader)):
		if i >= args.max_samples:
			break
		if torch.cuda.is_available():
			images = Variable(images.cuda(0))
			labels = Variable(labels.cuda(0))
		else:
			images = Variable(images)
			labels = Variable(labels)

		outputs = model(images)

		pred = np.squeeze((torch.max(outputs.data, 1, keepdim=True))[1].cpu().numpy())
		gt = np.squeeze(labels.data.cpu().numpy())

		for gt_, pred_ in zip(gt, pred):
			gts.append(gt_)
			preds.append(pred_)

	score, class_iou = scores(gts, preds, n_class=n_classes)

	for k, v in score.items():
		print(k, v)

	for i in range(n_classes):
		print(i, class_iou[i])


if __name__ == '__main__':
	validate()
