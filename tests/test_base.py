import torch

from cnn_finetune import make_model


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
