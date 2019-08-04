from torch import nn
from torchvision import models as torchvision_models

from cnn_finetune.base import ModelWrapperBase, ModelInfo
from cnn_finetune.utils import default


__all__ = [
    'ResNetWrapper', 'DenseNetWrapper', 'AlexNetWrapper', 'VGGWrapper',
    'SqueezeNetWrapper', 'InceptionV3Wrapper'
]


class TorchvisionWrapper(ModelWrapperBase):

    def get_original_model_info(self, original_model):
        return ModelInfo(
            input_space='RGB',
            input_size=[3, 224, 224],
            input_range=[0, 1],
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def get_original_model(self):
        model = getattr(torchvision_models, self.model_name)
        return model(pretrained=self.pretrained)

    def get_original_classifier(self, original_model):
        return original_model.classifier


class ResNetWrapper(TorchvisionWrapper):

    model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']

    def get_features(self, original_model):
        return nn.Sequential(*list(original_model.children())[:-2])

    def get_classifier_in_features(self, original_model):
        return original_model.fc.in_features

    def get_original_classifier(self, original_model):
        return original_model.fc


class DenseNetWrapper(TorchvisionWrapper):

    model_names = [
        'densenet121', 'densenet169', 'densenet201', 'densenet161'
    ]

    def get_features(self, original_model):
        return nn.Sequential(*original_model.features, nn.ReLU(inplace=True))

    def get_classifier_in_features(self, original_model):
        return original_model.classifier.in_features


class NetWithFcClassifierWrapper(TorchvisionWrapper):

    def check_args(
        self,
        model_name,
        pool,
        use_original_classifier,
        input_size,
        num_classes,
        pretrained,
        **kwargs
    ):
        super().check_args()
        if input_size is None:
            raise Exception(
                'You must provide input_size, e.g. '
                'make_model({model_name}, num_classes={num_classes}, '
                'pretrained={pretrained}, input_size=(224, 224)'.format(
                    model_name=model_name,
                    num_classes=num_classes,
                    pretrained=pretrained,
                )
            )

        if use_original_classifier:
            if pool is not None and pool is not default:
                raise Exception(
                    'You can\'t use pool layer with the original classifier'
                )
            if input_size != (224, 224):
                raise Exception(
                    'For the original classifier '
                    'input_size value must be (224, 224)'
                )

    def get_classifier_in_features(self, original_model):
        return self.calculate_classifier_in_features(original_model)

    def get_features(self, original_model):
        return original_model.features

    def get_pool(self):
        return None


class AlexNetWrapper(NetWithFcClassifierWrapper):

    model_names = ['alexnet']

    def get_classifier(self, in_features, num_classes):
        return nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )


class VGGWrapper(NetWithFcClassifierWrapper):

    model_names = [
        'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
        'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'
    ]

    def get_classifier(self, in_features, num_classes):
        return nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )


class SqueezeNetWrapper(TorchvisionWrapper):

    model_names = ['squeezenet1_0', 'squeezenet1_1']

    def get_features(self, original_model):
        return original_model.features

    def get_pool(self):
        return None

    def get_classifier_in_features(self, original_model):
        return self.calculate_classifier_in_features(original_model)

    def get_classifier(self, in_features, num_classes):
        classifier = nn.Sequential(
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        return classifier

    def forward(self, x):
        x = self.features(x)
        if self.pool is not None:
            x = self.pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)


class InceptionWrapper(ModelWrapperBase):
    # aux_logits and transform_input parameters from the torchvision
    # implementation are not supported

    def get_original_model_info(self, original_model):
        return ModelInfo(
            input_space='RGB',
            input_size=[3, 299, 299],
            input_range=[0, 1],
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )

    def get_original_model(self):
        model = getattr(torchvision_models, self.model_name)
        return model(pretrained=self.pretrained)

    def get_original_classifier(self, original_model):
        return original_model.fc

    def get_classifier_in_features(self, original_model):
        return original_model.fc.in_features


class InceptionV3Wrapper(InceptionWrapper):

    model_names = ['inception_v3']

    def get_features(self, original_model):
        features = nn.Sequential(
            original_model.Conv2d_1a_3x3,
            original_model.Conv2d_2a_3x3,
            original_model.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            original_model.Conv2d_3b_1x1,
            original_model.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            original_model.Mixed_5b,
            original_model.Mixed_5c,
            original_model.Mixed_5d,
            original_model.Mixed_6a,
            original_model.Mixed_6b,
            original_model.Mixed_6c,
            original_model.Mixed_6d,
            original_model.Mixed_6e,
            original_model.Mixed_7a,
            original_model.Mixed_7b,
            original_model.Mixed_7c,
        )
        return features


class GoogLeNetWrapper(InceptionWrapper):

    model_names = ['googlenet']

    def get_features(self, original_model):
        features = nn.Sequential(
            original_model.conv1,
            original_model.maxpool1,
            original_model.conv2,
            original_model.conv3,
            original_model.maxpool2,
            original_model.inception3a,
            original_model.inception3b,
            original_model.maxpool3,
            original_model.inception4a,
            original_model.inception4b,
            original_model.inception4c,
            original_model.inception4d,
            original_model.inception4e,
            original_model.maxpool4,
            original_model.inception5a,
            original_model.inception5b,
        )
        return features


class MobileNetV2Wrapper(TorchvisionWrapper):

    model_names = ['mobilenet_v2']

    def get_features(self, original_model):
        return original_model.features

    def get_original_classifier(self, original_model):
        return original_model.classifier[-1]

    def get_classifier_in_features(self, original_model):
        return original_model.classifier[-1].in_features


class ShuffleNetV2Wrapper(TorchvisionWrapper):

    model_names = ['shufflenet_v2_x0_5', 'shufflenet_v2_x1_0']

    def get_features(self, original_model):
        features = nn.Sequential(
            original_model.conv1,
            original_model.maxpool,
            original_model.stage2,
            original_model.stage3,
            original_model.stage4,
            original_model.conv5,
        )
        return features

    def get_original_classifier(self, original_model):
        return original_model.fc

    def get_classifier_in_features(self, original_model):
        return original_model.fc.in_features
