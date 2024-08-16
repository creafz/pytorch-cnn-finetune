from abc import ABCMeta, abstractmethod
from collections import namedtuple
import warnings

import torch
from torch import nn

from cnn_finetune.shims import no_grad_variable
from cnn_finetune.utils import default

ModelInfo = namedtuple('ModelInfo', ['input_space', 'input_size', 'input_range', 'mean', 'std'])

MODEL_REGISTRY = {}

class ModelRegistryMeta(type):
    """Metaclass that registers all model names in MODEL_REGISTRY."""

    def __new__(mcls, name, bases, namespace, **kwargs):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        model_names = namespace.get('model_names', [])
        for model_name in model_names:
            if model_name in MODEL_REGISTRY:
                prev_class = MODEL_REGISTRY[model_name]
                warnings.warn(f"Model name '{model_name}' already registered by {prev_class}")
            MODEL_REGISTRY[model_name] = cls
        return cls

class ModelWrapperMeta(ABCMeta, ModelRegistryMeta):
    """Metaclass combining ABCMeta and ModelRegistryMeta."""

class ModelWrapperBase(nn.Module, metaclass=ModelWrapperMeta):
    """Base class for all model wrappers."""

    flatten_features_output = True

    def __init__(self, *, model_name, num_classes, pretrained, dropout_p=None, pool=default,
                 classifier_factory=None, use_original_classifier=False, input_size=None,
                 original_model_state_dict=None, catch_output_size_exception=True):
        super().__init__()

        if num_classes < 1:
            raise ValueError('num_classes must be at least 1')
        if use_original_classifier and classifier_factory:
            raise ValueError('Cannot use classifier_factory with use_original_classifier=True')

        self.check_args(model_name=model_name, num_classes=num_classes, dropout_p=dropout_p, 
                        pretrained=pretrained, pool=pool, classifier_fn=classifier_factory, 
                        use_original_classifier=use_original_classifier, input_size=input_size)

        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.catch_output_size_exception = catch_output_size_exception

        original_model = self.get_original_model()
        if original_model_state_dict:
            original_model.load_state_dict(original_model_state_dict)

        self._features = self.get_features(original_model)
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p else None
        self.pool = self.get_pool() if pool is default else pool
        self.input_size = input_size

        if pretrained:
            self.original_model_info = self.get_original_model_info(original_model)
        else:
            self.original_model_info = None

        classifier_in_features = (self.calculate_classifier_in_features(original_model)
                                  if input_size else self.get_classifier_in_features(original_model))

        if use_original_classifier:
            self._classifier = self.get_original_classifier(original_model)
        else:
            self._classifier = (classifier_factory(classifier_in_features, num_classes)
                                if classifier_factory else self.get_classifier(classifier_in_features, num_classes))

    @abstractmethod
    def get_original_model(self):
        pass

    @abstractmethod
    def get_features(self, original_model):
        pass

    @abstractmethod
    def get_classifier_in_features(self, original_model):
        pass

    def get_original_model_info(self, original_model):
        return None

    def calculate_classifier_in_features(self, original_model):
        with no_grad_variable(torch.zeros(1, 3, *self.input_size)) as input_var:
            self.eval()
            try:
                output = self.features(input_var)
                if self.pool:
                    output = self.pool(output)
            except RuntimeError as e:
                if self.catch_output_size_exception and 'Output size is too small' in str(e):
                    message = (f'Input size {self.input_size} is too small for this model. '
                               'Increase input size and adjust the input_size argument.')
                    raise RuntimeError(message) from e
                raise
            self.train()
            return torch.numel(output[0])

    def check_args(self, **kwargs):
        pass

    def get_pool(self):
        return nn.AdaptiveAvgPool2d(1)

    def get_classifier(self, in_features, num_classes):
        return nn.Linear(in_features, self.num_classes)

    def get_original_classifier(self, original_model):
        raise NotImplementedError()

    def features(self, x):
        return self._features(x)

    def classifier(self, x):
        return self._classifier(x)

    def forward(self, x):
        x = self.features(x)
        if self.pool:
            x = self.pool(x)
        if self.dropout:
            x = self.dropout(x)
        if self.flatten_features_output:
            x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_model(model_name, num_classes, pretrained=True, dropout_p=None, pool=default,
               classifier_factory=None, use_original_classifier=False, input_size=None,
               original_model_state_dict=None, catch_output_size_exception=True):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model name '{model_name}' not found. Available: {', '.join(MODEL_REGISTRY.keys())}")
    wrapper = MODEL_REGISTRY[model_name]
    return wrapper(model_name=model_name, num_classes=num_classes, pretrained=pretrained, 
                   dropout_p=dropout_p, pool=pool, classifier_factory=classifier_factory, 
                   use_original_classifier=use_original_classifier, input_size=input_size, 
                   original_model_state_dict=original_model_state_dict, 
                   catch_output_size_exception=catch_output_size_exception)
