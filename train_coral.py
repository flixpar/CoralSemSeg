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

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.loss import cross_entropy2d
from ptsemseg.metrics import scores
from lr_scheduling import *

config = dict(
    img_size = 768,
    n_epoch = 100,
    batch_size = 2,
    learning_rate = 1e-5,
    feature_scale = 1,
)

def train(args):

    # Setup Dataloader
    print("Setting up dataloader...")
    data_loader = get_loader("coral")
    data_path = get_data_path("coral")
    loader = data_loader(data_path, img_size=args.img_size)

    n_classes = loader.n_classes
    n_channels = loader.n_channels

    trainloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=8, shuffle=True)
    print("Finished creating dataloader.")

    # Setup visdom for visualization
    vis = visdom.Visdom()

    loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='minibatches',
                                     ylabel='Loss',
                                     title='Training Loss',
                                     legend=['Loss']))

    # Setup Model
    print("Setting up model...")
    model = get_model("coralnet", n_classes, in_channels=n_channels)

    if torch.cuda.is_available():
        model.cuda(0)
        test_image, test_segmap = loader[0]
        test_image = Variable(test_image.unsqueeze(0).cuda(0))
    else:
        print("CUDA Error.")
        test_image, test_segmap = loader[0]
        test_image = Variable(test_image.unsqueeze(0))

    print("Setting up optimizer...")
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.99, weight_decay=5e-4)

    print("Starting training...")
    for epoch in range(args.n_epoch):
        print("Starting epoch {}".format(epoch))
        for i, (images, labels) in enumerate(trainloader):
            if torch.cuda.is_available():
                images = Variable(images.cuda(0))
                labels = Variable(labels.cuda(0))
            else:
                images = Variable(images)
                labels = Variable(labels)

            iter = len(trainloader)*epoch + i
            poly_lr_scheduler(optimizer, args.learning_rate, iter)

            optimizer.zero_grad()
            outputs = model(images)

            loss = cross_entropy2d(outputs, labels)

            loss.backward()
            optimizer.step()

            vis.line(
                X=torch.ones((1, 1)).cpu() * i,
                Y=torch.Tensor([loss.data[0]]).unsqueeze(0).cpu(),
                win=loss_window,
                update='append')

            if (i+1) % 20 == 0:
                print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, args.n_epoch, loss.data[0]))
                torch.save(model, "{}_{}_{}_{}.pkl".format("training/coralnet", "coral", args.feature_scale, epoch))



class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

if __name__ == '__main__':
    args = Namespace(**config)
    train(args)
