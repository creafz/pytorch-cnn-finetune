import types

import pytest
import torch.nn as nn
import pretrainedmodels

from cnn_finetune import make_model
from cnn_finetune.utils import default
from .utils import (
    assert_equal_model_outputs,
    assert_almost_equal_model_outputs,
    copy_module_weights
)


@pytest.mark.parametrize('model_name', ['resnext101_32x4d', 'resnext101_64x4d'])
@pytest.mark.parametrize(['pool', 'assert_equal_outputs'], [
    (nn.AvgPool2d((7, 7), (1, 1)), assert_equal_model_outputs),
    (default, assert_almost_equal_model_outputs),
])
def test_resnext_models(input_var, model_name, pool, assert_equal_outputs):
    original_model = getattr(pretrainedmodels, model_name)(
        pretrained='imagenet', num_classes=1000
    )
    finetune_model = make_model(
        model_name,
        num_classes=1000,
        pool=pool,
        pretrained=True,
    )
    copy_module_weights(original_model.last_linear, finetune_model._classifier)
    assert_equal_outputs(input_var, original_model, finetune_model)


@pytest.mark.parametrize('model_name', ['resnext101_32x4d', 'resnext101_64x4d'])
@pytest.mark.parametrize('input_var', [(1, 3, 256, 256)], indirect=True)
def test_resnext_models_with_another_input_size(input_var, model_name):
    model = make_model(model_name, num_classes=1000, pretrained=True)
    model(input_var)


@pytest.mark.parametrize(['pool', 'assert_equal_outputs'], [
    (nn.AvgPool2d(11, stride=1, padding=0), assert_equal_model_outputs),
    (default, assert_almost_equal_model_outputs),
])
@pytest.mark.parametrize('input_var', [(1, 3, 331, 331)], indirect=True)
def test_nasnetalarge_model(input_var, pool, assert_equal_outputs):
    original_model = pretrainedmodels.nasnetalarge(
        pretrained='imagenet', num_classes=1000
    )
    finetune_model = make_model(
        'nasnetalarge',
        num_classes=1000,
        pool=pool,
        pretrained=True,
    )
    copy_module_weights(original_model.last_linear, finetune_model._classifier)
    assert_equal_outputs(input_var, original_model, finetune_model)


@pytest.mark.parametrize('input_var', [(1, 3, 256, 256)], indirect=True)
def test_nasnetalarge_model_with_another_input_size(input_var):
    model = make_model('nasnetalarge', num_classes=1000, pretrained=True)
    model(input_var)


@pytest.mark.parametrize(['pool', 'assert_equal_outputs'], [
    (nn.AvgPool2d(7, stride=1, padding=0), assert_equal_model_outputs),
    (default, assert_almost_equal_model_outputs),
])
@pytest.mark.parametrize('input_var', [(1, 3, 224, 224)], indirect=True)
def test_nasnetamobile_model(input_var, pool, assert_equal_outputs):
    original_model = pretrainedmodels.nasnetamobile(
        pretrained='imagenet', num_classes=1000
    )
    finetune_model = make_model(
        'nasnetamobile',
        num_classes=1000,
        pool=pool,
        pretrained=True,
    )
    copy_module_weights(original_model.last_linear, finetune_model._classifier)
    assert_equal_outputs(input_var, original_model, finetune_model)


@pytest.mark.parametrize('input_var', [(1, 3, 256, 256)], indirect=True)
def test_nasnetamobile_model_with_another_input_size(input_var):
    model = make_model('nasnetamobile', num_classes=1000, pretrained=True)
    model(input_var)


@pytest.mark.parametrize(['pool', 'assert_equal_outputs'], [
    (nn.AvgPool2d(8, count_include_pad=False), assert_equal_model_outputs),
    (default, assert_almost_equal_model_outputs),
])
@pytest.mark.parametrize('input_var', [(1, 3, 299, 299)], indirect=True)
def test_inceptionresnetv2_model(input_var, pool, assert_equal_outputs):
    original_model = pretrainedmodels.inceptionresnetv2(
        pretrained='imagenet', num_classes=1000
    )
    finetune_model = make_model(
        'inceptionresnetv2',
        num_classes=1000,
        pool=pool,
        pretrained=True,
    )
    copy_module_weights(original_model.last_linear, finetune_model._classifier)
    assert_equal_outputs(input_var, original_model, finetune_model)


@pytest.mark.parametrize('input_var', [(1, 3, 256, 256)], indirect=True)
def test_inceptionresnetv2_model_with_another_input_size(input_var):
    model = make_model('inceptionresnetv2', num_classes=1000, pretrained=True)
    model(input_var)


@pytest.mark.parametrize(
    'model_name',
    ['dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn131', 'dpn107']
)
@pytest.mark.parametrize(['pool', 'assert_equal_outputs'], [
    (nn.AvgPool2d(kernel_size=7, stride=1), assert_equal_model_outputs),
    (default, assert_almost_equal_model_outputs),
])
def test_dpn_models(input_var, model_name, pool, assert_equal_outputs):
    if model_name in {'dpn68b', 'dpn92', 'dpn107'}:
        pretrained = 'imagenet+5k'
    else:
        pretrained = 'imagenet'
    original_model = getattr(pretrainedmodels, model_name)(
        pretrained=pretrained, num_classes=1000
    )
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
    ['dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn131', 'dpn107']
)
@pytest.mark.parametrize('input_var', [(1, 3, 256, 256)], indirect=True)
def test_dpn_models_with_another_input_size(model_name, input_var):
    model = make_model(model_name, num_classes=1000, pretrained=True)
    model(input_var)


@pytest.mark.parametrize(['pool', 'assert_equal_outputs'], [
    (nn.AvgPool2d(8, count_include_pad=False), assert_equal_model_outputs),
    (default, assert_almost_equal_model_outputs),
])
@pytest.mark.parametrize('input_var', [(1, 3, 299, 299)], indirect=True)
def test_inceptionv4_model(input_var, pool, assert_equal_outputs):
    original_model = pretrainedmodels.inceptionv4(
        pretrained='imagenet', num_classes=1000
    )
    finetune_model = make_model(
        'inception_v4',
        num_classes=1000,
        pool=pool,
        pretrained=True,
    )
    copy_module_weights(original_model.last_linear, finetune_model._classifier)
    assert_equal_outputs(input_var, original_model, finetune_model)


@pytest.mark.parametrize('input_var', [(1, 3, 256, 256)], indirect=True)
def test_inceptionv4_model_with_another_input_size(input_var):
    model = make_model('inception_v4', num_classes=1000, pretrained=True)
    model(input_var)


@pytest.mark.skip(
    'Xception model fails to load in PyTorch 0.4. '
    'https://github.com/Cadene/pretrained-models.pytorch/issues/62'
)
@pytest.mark.parametrize('input_var', [(1, 3, 256, 256)], indirect=True)
def test_xception_model_with_another_input_size(input_var):
    model = make_model('xception', num_classes=1000, pretrained=True)
    model(input_var)


@pytest.mark.parametrize(
    'model_name',
    [
        'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152',
        'se_resnext50_32x4d', 'se_resnext101_32x4d',
    ]
)
@pytest.mark.parametrize(['pool', 'assert_equal_outputs'], [
    (nn.AvgPool2d((7, 7), (1, 1)), assert_equal_model_outputs),
    (default, assert_almost_equal_model_outputs),
])
def test_senet_models(input_var, model_name, pool, assert_equal_outputs):
    original_model = getattr(pretrainedmodels, model_name)(
        pretrained='imagenet', num_classes=1000
    )
    finetune_model = make_model(
        model_name,
        num_classes=1000,
        pool=pool,
        pretrained=True,
    )
    copy_module_weights(original_model.last_linear, finetune_model._classifier)
    assert_equal_outputs(input_var, original_model, finetune_model)


@pytest.mark.parametrize(
    'model_name',
    [
        'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152',
        'se_resnext50_32x4d', 'se_resnext101_32x4d',
    ]
)
@pytest.mark.parametrize('input_var', [(1, 3, 256, 256)], indirect=True)
def test_senet_models_with_another_input_size(input_var, model_name):
    model = make_model(model_name, num_classes=1000, pretrained=True)
    model(input_var)


@pytest.mark.parametrize(['pool', 'assert_equal_outputs'], [
    (nn.AvgPool2d(11, stride=1, padding=0), assert_equal_model_outputs),
    (default, assert_almost_equal_model_outputs),
])
@pytest.mark.parametrize('input_var', [(1, 3, 331, 331)], indirect=True)
def test_pnasnet5large_model(input_var, pool, assert_equal_outputs):
    original_model = pretrainedmodels.pnasnet5large(
        pretrained='imagenet', num_classes=1000
    )
    finetune_model = make_model(
        'pnasnet5large',
        num_classes=1000,
        pool=pool,
        pretrained=True,
    )
    copy_module_weights(original_model.last_linear, finetune_model._classifier)
    assert_equal_outputs(input_var, original_model, finetune_model)


@pytest.mark.parametrize('input_var', [(1, 3, 256, 256)], indirect=True)
def test_pnasnet5large_model_with_another_input_size(input_var):
    model = make_model('pnasnet5large', num_classes=1000, pretrained=True)
    model(input_var)


@pytest.mark.parametrize(['pool', 'assert_equal_outputs'], [
    (nn.AvgPool2d(9, stride=1), assert_equal_model_outputs),
    (default, assert_almost_equal_model_outputs),
])
@pytest.mark.parametrize('input_var', [(1, 3, 331, 331)], indirect=True)
def test_polynet_model(input_var, pool, assert_equal_outputs):
    original_model = pretrainedmodels.polynet(
        pretrained='imagenet', num_classes=1000
    )
    finetune_model = make_model(
        'polynet',
        num_classes=1000,
        pool=pool,
        pretrained=True,
    )
    copy_module_weights(original_model.last_linear, finetune_model._classifier)
    assert_equal_outputs(input_var, original_model, finetune_model)


@pytest.mark.parametrize('input_var', [(1, 3, 256, 256)], indirect=True)
def test_polynet_model_with_another_input_size(input_var):
    model = make_model('polynet', num_classes=1000, pretrained=True)
    model(input_var)
