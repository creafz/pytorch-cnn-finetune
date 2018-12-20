### Fine-tune pretrained Convolutional Neural Networks with PyTorch.


[![PyPI](https://img.shields.io/pypi/v/cnn-finetune.svg)](https://pypi.org/project/cnn-finetune/)
[![CircleCI](https://circleci.com/gh/creafz/pytorch-cnn-finetune/tree/master.svg?style=shield)](https://circleci.com/gh/creafz/pytorch-cnn-finetune)
[![codecov.io](https://codecov.io/github/creafz/pytorch-cnn-finetune/coverage.svg?branch=master)](https://codecov.io/github/creafz/pytorch-cnn-finetune)


### Features
- Gives access to the most popular CNN architectures pretrained on ImageNet.
- Automatically replaces classifier on top of the network, which allows you to train a network with a dataset that has a different number of classes.
- Allows you to use images with any resolution (and not only the resolution that was used for training the original model on ImageNet).
- Allows adding a Dropout layer or a custom pooling layer.


### Supported architectures and models

#### From [torchvision](https://github.com/pytorch/vision/) package:

- ResNet (`resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`)
- DenseNet (`densenet121`, `densenet169`, `densenet201`, `densenet161`)
- Inception v3 (`inception_v3`)
- VGG (`vgg11`, `vgg11_bn`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19`, `vgg19_bn`)
- SqueezeNet (`squeezenet1_0`, `squeezenet1_1`)
- AlexNet (`alexnet`)

#### From [Pretrained models for PyTorch](https://github.com/Cadene/pretrained-models.pytorch) package:
- ResNeXt (`resnext101_32x4d`, `resnext101_64x4d`)
- NASNet-A Large (`nasnetalarge`)
- NASNet-A Mobile (`nasnetamobile`)
- Inception-ResNet v2 (`inceptionresnetv2`)
- Dual Path Networks (`dpn68`, `dpn68b`, `dpn92`, `dpn98`, `dpn131`, `dpn107`)
- Inception v4 (`inception_v4`)
- Xception (`xception`)
- Squeeze-and-Excitation Networks (`senet154`, `se_resnet50`, `se_resnet101`, `se_resnet152`, `se_resnext50_32x4d`, `se_resnext101_32x4d`)
- PNASNet-5-Large (`pnasnet5large`)
- PolyNet (`polynet`)


### Requirements
* Python 3.5+
* PyTorch 0.3+

### Installation

```
pip install cnn_finetune
```


### Major changes:
#### Version 0.4
- Default value for `pretrained` argument in `make_model` is changed from `False` to `True`. Now call `make_model('resnet18', num_classes=10)` is equal to `make_model('resnet18', num_classes=10, pretrained=True)`


### Example usage:

#### Make a model with ImageNet weights for 10 classes

```
from cnn_finetune import make_model

model = make_model('resnet18', num_classes=10, pretrained=True)
```

#### Make a model with Dropout
```
model = make_model('nasnetalarge', num_classes=10, pretrained=True, dropout_p=0.5)
```

#### Make a model with Global Max Pooling instead of Global Average Pooling
```
import torch.nn as nn

model = make_model('inceptionresnetv2', num_classes=10, pretrained=True, pool=nn.AdaptiveMaxPool2d(1))
```


#### Make a VGG16 model that takes images of size 256x256 pixels
VGG and AlexNet models use fully-connected layers, so you have to additionally pass the input size of images
when constructing a new model. This information is needed to determine the input size of fully-connected layers.
```
model = make_model('vgg16', num_classes=10, pretrained=True, input_size=(256, 256))
```


#### Make a VGG16 model that takes images of size 256x256 pixels and uses a custom classifier
```
import torch.nn as nn

def make_classifier(in_features, num_classes):
    return nn.Sequential(
        nn.Linear(in_features, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, num_classes),
    )

model = make_model('vgg16', num_classes=10, pretrained=True, input_size=(256, 256), classifier_factory=make_classifier)
```


#### Show preprocessing that was used to train the original model on ImageNet
```
>> model = make_model('resnext101_64x4d', num_classes=10, pretrained=True)
>> print(model.original_model_info)
ModelInfo(input_space='RGB', input_size=[3, 224, 224], input_range=[0, 1], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
>> print(model.original_model_info.mean)
[0.485, 0.456, 0.406]
```

#### CIFAR10 Example
See [examples/cifar10.py](examples/cifar10.py) file (requires PyTorch 0.4+).
