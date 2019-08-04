from multiprocessing.pool import Pool

import pytest
import torch

from cnn_finetune import make_model
from cnn_finetune.base import MODEL_REGISTRY
from .utils import (
    assert_iterable_length_and_type,
    get_default_input_size_for_model,
)


def make_model_x(model_name):
    model = make_model(model_name, pretrained=True, num_classes=10, input_size=(224, 224))


def test_load_state_dict():

    p = Pool(128)
    print(p.map( make_model_x, MODEL_REGISTRY.keys()))

