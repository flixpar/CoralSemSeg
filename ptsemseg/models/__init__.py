import torchvision.models as models

from ptsemseg.models.fcn import *
from ptsemseg.models.segnet import *
from ptsemseg.models.unet import *
from ptsemseg.models.pspnet import *
from ptsemseg.models.linknet import *
from ptsemseg.models.coralnet import *
from ptsemseg.models.segnet_mangrove import *

def get_model(name, n_classes, in_channels=3):
    model = _get_model_instance(name)

    if name in ['fcn32s', 'fcn16s', 'fcn8s']:
        model = model(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == 'segnet':
        model = model(n_classes=n_classes,
                      is_unpooling=True)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == 'unet':
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=3,
                      is_deconv=True)

    elif name == 'linknet':
        model = model(n_classes=n_classes,
                      resnet='resnet50',
                      pretrained=True,
                      in_channels=3)

    elif name == 'coralnet':
        model = model(n_classes=n_classes,
                     resnet='resnet101',
                     pretrained=False,
                     in_channels=in_channels)

    elif name == 'segnet_mangrove':
        model = model(n_classes=n_classes, in_channels=in_channels)

    else:
        raise 'Model {} not available'.format(name)

    return model

def _get_model_instance(name):
    return {
        'fcn32s': fcn32s,
        'fcn8s': fcn8s,
        'fcn16s': fcn16s,
        'unet': unet,
        'segnet': segnet,
        'pspnet': pspnet,
        'linknet': linknet,
        'coralnet': coralnet,
        'segnet_mangrove': segnet_mangrove,
    }[name]
