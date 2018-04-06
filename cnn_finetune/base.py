from abc import ABCMeta, abstractmethod
from collections import namedtuple
import warnings

import torch
from torch import nn

from cnn_finetune.shims import no_grad_variable
from cnn_finetune.utils import default, product


ModelInfo = namedtuple(
    'ModelInfo',
    ['input_space', 'input_size', 'input_range', 'mean', 'std']
)

# Global registry which is used to track wrappers for all model names.
MODEL_REGISTRY = {}


class ModelRegistryMeta(type):
    """Metaclass that registers all model names defined in model_names property
    of a descendant class in the global MODEL_REGISTRY.
    """

    def __new__(mcls, name, bases, namespace, **kwargs):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        if 'model_names' in namespace:
            for model_name in namespace['model_names']:
                # If model_name is already registered,
                # override the wrapper definition and display a warning.
                if model_name in MODEL_REGISTRY:
                    current_class = "<class '{module}.{qualname}'>".format(
                        module=namespace['__module__'],
                        qualname=namespace['__qualname__'],
                    )
                    warnings.warn(
                        "{current_class} redefined model_name '{model_name}'"
                        "that was already registered by "
                        "{previous_class}".format(
                            current_class=current_class,
                            model_name=model_name,
                            previous_class=MODEL_REGISTRY[model_name]
                        )
                    )
                MODEL_REGISTRY[model_name] = cls
        return cls


class ModelWrapperMeta(ABCMeta, ModelRegistryMeta):
    """An intermediate class that allows usage of both
    ABCMeta and ModelRegistryMeta simultaneously
    """
    pass


class ModelWrapperBase(nn.Module, metaclass=ModelWrapperMeta):
    """Base class for all wrappers. To create a new wrapper you should
    subclass it and add model names that are supported by the wrapper to
    the model_names property. Those model names will be automatically
    registered in the global MODEL_REGISTRY upon class initialization.
    """

    # If True an output of .features() call will be converted
    # to a tensor of shape [B, C * H * W].
    flatten_features_output = True

    def __init__(self, *, model_name, num_classes, pretrained, dropout_p, pool,
                 classifier_factory, use_original_classifier, input_size,
                 original_model_state_dict):
        super().__init__()

        if num_classes < 1:
            raise ValueError('num_classes should be greater or equal to 1')

        if use_original_classifier and classifier_factory:
            raise ValueError(
                'You can\'t use classifier_factory when '
                'use_original_classifier is set to True'
            )
        self.check_args(
            model_name=model_name,
            num_classes=num_classes,
            dropout_p=dropout_p,
            pretrained=pretrained,
            pool=pool,
            classifier_fn=classifier_factory,
            use_original_classifier=use_original_classifier,
            input_size=input_size,
        )

        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained

        original_model = self.get_original_model()
        if original_model_state_dict is not None:
            original_model.load_state_dict(original_model_state_dict)

        self._features = self.get_features(original_model)
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p else None
        self.pool = self.get_pool() if pool is default else pool
        self.input_size = input_size

        if pretrained:
            self.original_model_info = self.get_original_model_info(
                original_model
            )
        else:
            self.original_model_info = None

        if input_size:
            classifier_in_features = self.calculate_classifier_in_features(
                original_model
            )
        else:
            classifier_in_features = self.get_classifier_in_features(
                original_model
            )

        if use_original_classifier:
            classifier = self.get_original_classifier(original_model)
        else:
            if classifier_factory:
                classifier = classifier_factory(
                    classifier_in_features, num_classes,
                )
            else:
                classifier = self.get_classifier(
                    classifier_in_features, num_classes
                )
        self._classifier = classifier

    @abstractmethod
    def get_original_model(self):
        # Should return a model that will be later passed to
        # methods that will construct a model for fine-tuning.
        pass

    @abstractmethod
    def get_features(self, original_model):
        # Should return an instance of nn.Module that will be used as
        # a feature extractor.
        pass

    @abstractmethod
    def get_classifier_in_features(self, original_model):
        # Should return a number of input features for classifier
        # for a case when default pooling layer is being used.
        pass

    def get_original_model_info(self, original_model):
        # Should return an instance of ModelInfo.
        return None

    def calculate_classifier_in_features(self, original_model):
        # Runs forward pass through feature extractor to get
        # the number of input features for classifier.

        with no_grad_variable(torch.zeros(1, 3, *self.input_size)) as input_var:
            # Set model to the eval mode so forward pass
            # won't affect BatchNorm statistics.
            original_model.eval()
            output = original_model.features(input_var)
            if self.pool is not None:
                output = self.pool(output)
            original_model.train()
            return product(output.size()[1:])

    def check_args(self, **kwargs):
        # Allows additional arguments checking by model wrappers.
        pass

    def get_pool(self):
        # Returns default pooling layer for model. May return None to
        # indicate absence of pooling layer in a model.
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
        if self.pool is not None:
            x = self.pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.flatten_features_output:
            x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_model(
    model_name,
    num_classes,
    pretrained=False,
    dropout_p=None,
    pool=default,
    classifier_factory=None,
    use_original_classifier=False,
    input_size=None,
    original_model_state_dict=None,
):
    """
    Args:
        model_name (str): Name of the model.
        num_classes (int): Number of classes for the classifier.
        pretrained (bool, optional) If True uses ImageNet weights for the
            original model.
        dropout_p (float, optional) Dropout probability.
        pool (nn.Module or None, optional) Custom pooling layer.
        classifier_factory (callable, optional) Allows creating a custom
            classifier instead of using nn.Linear. Should be a callable
            that takes the number of input features and the number of classes
            as arguments and returns a classifier module.
        use_original_classifier  (bool, optional) If True uses classifier from
            the original model.
        input_size (tuple, optional) Input size of  images that will be
            fed into the network. Should be a tuple containing (width, height)
            in pixels. Required for architectures that use fully-connected
            layers such as AlexNet or VGG.
        original_model_state_dict (dict, optional): Dict containing
            parameters for the original model.
    """

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            'model_name {model_name} not found. '
            'Available model_name values: {model_names}'.format(
                model_name=model_name,
                model_names=', '.join(MODEL_REGISTRY.keys())
            )
        )
    wrapper = MODEL_REGISTRY[model_name]
    return wrapper(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_p=dropout_p,
        pool=pool,
        classifier_factory=classifier_factory,
        use_original_classifier=use_original_classifier,
        input_size=input_size,
        original_model_state_dict=original_model_state_dict,
    )
