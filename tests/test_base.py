import pytest
import torch

from cnn_finetune import make_model
from cnn_finetune.base import MODEL_REGISTRY
from .utils import assert_iterable_length_and_type


def test_load_state_dict():
    torch.manual_seed(42)
    model1 = make_model('resnet18', num_classes=10)
    model1_state = model1.state_dict()

    torch.manual_seed(84)
    model2 = make_model('resnet18', num_classes=10)
    model2_state = model2.state_dict()

    assert not all(
        torch.equal(weights1, weights2) for weights1, weights2
        in zip(model1_state.values(), model2_state.values())
    )

    model2.load_state_dict(model1_state)
    model2_state = model2.state_dict()

    assert all(
        torch.equal(weights1, weights2) for weights1, weights2 in
        zip(model1_state.values(), model2_state.values())
    )


def test_state_dict_features_and_classifier():
    model = make_model('resnet18', num_classes=10)
    model_state_keys = model.state_dict().keys()

    assert '_classifier.weight' in model_state_keys
    assert '_classifier.bias' in model_state_keys

    features_keys = [
        key for key in model_state_keys if key.startswith('_features')
    ]
    assert len(features_keys) == 100


@pytest.mark.parametrize('model_name', list(MODEL_REGISTRY.keys()))
def test_every_pretrained_model_has_model_info(model_name):
    input_size = None
    if (
        model_name == 'alexnet'
        or model_name.startswith('vgg')
        or model_name.startswith('squeezenet')
    ):
        input_size = (224, 224)
    model = make_model(
        model_name,
        pretrained=True,
        num_classes=10,
        dropout_p=0.5,
        input_size=input_size,
    )
    original_model_info = model.original_model_info
    assert original_model_info

    assert_iterable_length_and_type(
        original_model_info.input_space, 3, str
    )
    assert_iterable_length_and_type(
        original_model_info.input_size, 3, int
    )
    assert_iterable_length_and_type(
        original_model_info.input_range, 2, (int, float)
    )
    assert_iterable_length_and_type(
        original_model_info.mean, 3, (int, float)
    )
    assert_iterable_length_and_type(
        original_model_info.std, 3, (int, float)
    )


@pytest.mark.parametrize('model_name', list(MODEL_REGISTRY.keys()))
def test_models_without_pretrained_weights_dont_have_model_info(model_name):
    input_size = None
    if (
        model_name == 'alexnet'
        or model_name.startswith('vgg')
        or model_name.startswith('squeezenet')
    ):
        input_size = (224, 224)
    model = make_model(
        model_name,
        pretrained=False,
        num_classes=10,
        dropout_p=0.5,
        input_size=input_size,
    )
    assert model.original_model_info is None


@pytest.mark.parametrize('model_name', list(MODEL_REGISTRY.keys()))
@pytest.mark.parametrize('pretrained', [True, False])
def test_make_model_with_specific_input_size(model_name, pretrained):
    make_model(
        model_name,
        pretrained=pretrained,
        num_classes=10,
        dropout_p=0.5,
        input_size=(256, 256),
    )


def test_make_model_error_message_for_small_input_size():
    expected_message_end = (
        'Input size (8, 8) is too small for this model. Try increasing '
        'the input size of images and change the value of input_size '
        'argument accordingly.'
    )
    with pytest.raises(RuntimeError) as exc_info:
        make_model('vgg11', pretrained=True, num_classes=10, input_size=(8, 8))
    assert str(exc_info.value).endswith(expected_message_end)


def test_make_model_error_message_for_small_input_size_without_catching_exc():
    unexpected_message_end = (
        'Input size (8, 8) is too small for this model. Try increasing '
        'the input size of images and change the value of input_size '
        'argument accordingly.'
    )
    with pytest.raises(RuntimeError) as exc_info:
        make_model(
            'vgg11',
            pretrained=True,
            num_classes=10,
            input_size=(8, 8),
            catch_output_size_exception=False,
        )
    assert not str(exc_info.value).endswith(unexpected_message_end)
