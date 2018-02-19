import pytest
import torch.nn as nn
from torchvision import models as torchvision_models

from cnn_finetune import make_model
from cnn_finetune.utils import default
from .utils import (
    assert_equal_model_outputs,
    assert_almost_equal_model_outputs,
    copy_module_weights
)


@pytest.mark.parametrize(
    'model_name',
    ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
)
@pytest.mark.parametrize(['pool', 'assert_equal_outputs'], [
    (nn.AvgPool2d(7, stride=1), assert_equal_model_outputs),
    (default, assert_almost_equal_model_outputs),
])
def test_resnet_models(
    input_var, model_name, pool, assert_equal_outputs
):
    original_model = getattr(torchvision_models, model_name)(pretrained=True)
    finetune_model = make_model(
        model_name,
        num_classes=1000,
        pool=pool,
        pretrained=True,
    )
    copy_module_weights(original_model.fc, finetune_model._classifier)
    assert_equal_outputs(input_var, original_model, finetune_model)


@pytest.mark.parametrize(
    'model_name',
    ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
)
@pytest.mark.parametrize('input_var', [(1, 3, 256, 256)], indirect=True)
def test_resnet_models_with_another_input_size(input_var, model_name):
    model = make_model(model_name, num_classes=1000, pretrained=True)
    model(input_var)


@pytest.mark.parametrize(
    'model_name',
    ['densenet121', 'densenet169', 'densenet201',  'densenet161']
)
@pytest.mark.parametrize(['pool', 'assert_equal_outputs'], [
    (nn.AvgPool2d(7, stride=1), assert_equal_model_outputs),
    (default, assert_almost_equal_model_outputs),
])
def test_densenet_models(input_var, model_name, pool, assert_equal_outputs):
    original_model = getattr(torchvision_models, model_name)(pretrained=True)
    finetune_model = make_model(
        model_name,
        num_classes=1000,
        pool=pool,
        pretrained=True,
    )
    copy_module_weights(original_model.classifier, finetune_model._classifier)
    assert_equal_outputs(input_var, original_model, finetune_model)


@pytest.mark.parametrize(
    'model_name',
    ['densenet121', 'densenet169', 'densenet201',  'densenet161']
)
@pytest.mark.parametrize('input_var', [(1, 3, 256, 256)], indirect=True)
def test_densenet_models_with_another_input_size(input_var, model_name):
    model = make_model(model_name, num_classes=1000, pretrained=True)
    model(input_var)


def test_alexnet_model_with_default_classifier(input_var):
    original_model = torchvision_models.alexnet(pretrained=True)
    original_model(input_var)
    finetune_model = make_model(
        'alexnet',
        num_classes=1000,
        use_original_classifier=True,
        input_size=(224, 224),
        pretrained=True,
    )
    assert_equal_model_outputs(input_var, original_model, finetune_model)


@pytest.mark.parametrize('pool', [default, nn.AdaptiveAvgPool2d(1)])
@pytest.mark.parametrize('input_var', [(1, 3, 256, 256)], indirect=True)
def test_alexnet_model_with_another_input_size(input_var, pool):
    model = make_model(
        'alexnet',
        num_classes=1000,
        input_size=(256, 256),
        pool=pool,
        pretrained=True,
    )
    model(input_var)


@pytest.mark.parametrize(
    'model_name',
    [
        'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
        'vgg19_bn', 'vgg19'
    ]
)
def test_vgg_models_with_default_classifier(model_name, input_var):
    original_model = getattr(torchvision_models, model_name)(pretrained=True)
    finetune_model = make_model(
        model_name,
        num_classes=1000,
        use_original_classifier=True,
        input_size=(224, 224),
        pretrained=True,
    )
    assert_equal_model_outputs(input_var, original_model, finetune_model)


@pytest.mark.parametrize(
    'model_name',
    [
        'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
        'vgg19_bn', 'vgg19'
    ]
)
@pytest.mark.parametrize('pool', [default, nn.AdaptiveAvgPool2d(1)])
@pytest.mark.parametrize('input_var', [(1, 3, 256, 256)], indirect=True)
def test_vgg_models_with_another_input_size(model_name, input_var, pool):
    model = make_model(
        model_name,
        num_classes=1000,
        input_size=(256, 256),
        pool=pool,
        pretrained=True,
    )
    model(input_var)


@pytest.mark.parametrize('model_name', ['squeezenet1_0', 'squeezenet1_1'])
def test_squeezenet_models_with_original_classifier(model_name, input_var):
    original_model = getattr(torchvision_models, model_name)(pretrained=True)
    finetune_model = make_model(
        model_name,
        num_classes=1000,
        use_original_classifier=True,
        input_size=(224, 224),
        pretrained=True,
    )
    assert_equal_model_outputs(input_var, original_model, finetune_model)


@pytest.mark.parametrize('model_name', ['squeezenet1_0', 'squeezenet1_1'])
@pytest.mark.parametrize('pool', [default, nn.AdaptiveAvgPool2d(1)])
@pytest.mark.parametrize('input_var', [(1, 3, 256, 256)], indirect=True)
def test_squeezenet_models_with_another_input_size(model_name, input_var, pool):
    model = make_model(
        model_name,
        num_classes=1000,
        input_size=(256, 256),
        pool=pool,
        pretrained=True,
    )
    model(input_var)


@pytest.mark.parametrize(['pool', 'assert_equal_outputs'], [
    (nn.AvgPool2d(kernel_size=8), assert_equal_model_outputs),
    (default, assert_almost_equal_model_outputs),
])
@pytest.mark.parametrize('input_var', [(1, 3, 299, 299)], indirect=True)
def test_inception_v3_model(input_var, pool, assert_equal_outputs):
    original_model = torchvision_models.inception_v3(
        pretrained=True,
        transform_input=False,
    )
    finetune_model = make_model(
        'inception_v3', num_classes=1000, pool=pool, pretrained=True
    )
    copy_module_weights(original_model.fc, finetune_model._classifier)
    assert_equal_outputs(input_var, original_model, finetune_model)


@pytest.mark.parametrize('input_var', [(1, 3, 350, 350)], indirect=True)
def test_inception_v3_model_with_another_input_size(input_var):
    model = make_model('inception_v3', num_classes=1000, pretrained=True)
    model(input_var)
