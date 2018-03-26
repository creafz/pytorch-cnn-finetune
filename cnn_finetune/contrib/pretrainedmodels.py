from torch import nn
import pretrainedmodels
from pretrainedmodels.models.dpn import adaptive_avgmax_pool2d

from cnn_finetune.base import ModelWrapperBase, ModelInfo


__all__ = [
    'ResNeXtWrapper', 'NasNetWrapper', 'InceptionResNetV2Wrapper',
    'DPNWrapper', 'InceptionV4Wrapper', 'XceptionWrapper',
    'NasNetMobileWrapper'
]


class PretrainedModelsWrapper(ModelWrapperBase):

    def get_original_model_info(self, original_model):
        return ModelInfo(
            input_space=original_model.input_space,
            input_size=original_model.input_size,
            input_range=original_model.input_range,
            mean=original_model.mean,
            std=original_model.std,
        )

    def get_original_model(self):
        model = getattr(pretrainedmodels, self.model_name)
        if self.pretrained:
            model_kwargs = {'pretrained': 'imagenet', 'num_classes': 1000}
        else:
            model_kwargs = {'pretrained': None}
        return model(**model_kwargs)

    def get_features(self, original_model):
        return original_model.features

    def get_original_classifier(self, original_model):
        return original_model.last_linear

    def get_classifier_in_features(self, original_model):
        return original_model.last_linear.in_features


class ResNeXtWrapper(PretrainedModelsWrapper):

    model_names = ['resnext101_32x4d', 'resnext101_64x4d']


class NasNetWrapper(PretrainedModelsWrapper):

    model_names = ['nasnetalarge']

    def get_features(self, original_model):
        features = nn.Module()
        for name, module in list(original_model.named_children())[:-3]:
            features.add_module(name, module)
        return features

    def features(self, x):
        x_conv0 = self._features.conv0(x)
        x_stem_0 = self._features.cell_stem_0(x_conv0)
        x_stem_1 = self._features.cell_stem_1(x_conv0, x_stem_0)

        x_cell_0 = self._features.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self._features.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self._features.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self._features.cell_3(x_cell_2, x_cell_1)
        x_cell_4 = self._features.cell_4(x_cell_3, x_cell_2)
        x_cell_5 = self._features.cell_5(x_cell_4, x_cell_3)

        x_reduction_cell_0 = self._features.reduction_cell_0(
            x_cell_5, x_cell_4
        )

        x_cell_6 = self._features.cell_6(x_reduction_cell_0, x_cell_4)
        x_cell_7 = self._features.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self._features.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self._features.cell_9(x_cell_8, x_cell_7)
        x_cell_10 = self._features.cell_10(x_cell_9, x_cell_8)
        x_cell_11 = self._features.cell_11(x_cell_10, x_cell_9)

        x_reduction_cell_1 = self._features.reduction_cell_1(
            x_cell_11, x_cell_10
        )

        x_cell_12 = self._features.cell_12(x_reduction_cell_1, x_cell_10)
        x_cell_13 = self._features.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self._features.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self._features.cell_15(x_cell_14, x_cell_13)
        x_cell_16 = self._features.cell_16(x_cell_15, x_cell_14)
        x_cell_17 = self._features.cell_17(x_cell_16, x_cell_15)
        x = self._features.relu(x_cell_17)
        return x


class NasNetMobileWrapper(PretrainedModelsWrapper):

    model_names = ['nasnetamobile']

    def get_features(self, original_model):
        features = nn.Module()
        for name, module in list(original_model.named_children())[:-3]:
            features.add_module(name, module)
        return features

    def features(self, input):
        x_conv0 = self._features.conv0(input)
        x_stem_0 = self._features.cell_stem_0(x_conv0)
        x_stem_1 = self._features.cell_stem_1(x_conv0, x_stem_0)

        x_cell_0 = self._features.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self._features.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self._features.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self._features.cell_3(x_cell_2, x_cell_1)

        x_reduction_cell_0 = self._features.reduction_cell_0(x_cell_3, x_cell_2)

        x_cell_6 = self._features.cell_6(x_reduction_cell_0, x_cell_3)
        x_cell_7 = self._features.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self._features.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self._features.cell_9(x_cell_8, x_cell_7)

        x_reduction_cell_1 = self._features.reduction_cell_1(x_cell_9, x_cell_8)

        x_cell_12 = self._features.cell_12(x_reduction_cell_1, x_cell_9)
        x_cell_13 = self._features.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self._features.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self._features.cell_15(x_cell_14, x_cell_13)
        x = self._features.relu(x_cell_15)
        return x


class InceptionResNetV2Wrapper(PretrainedModelsWrapper):

    model_names = ['inceptionresnetv2']

    def get_features(self, original_model):
        return nn.Sequential(*list(original_model.children())[:-2])


class DPNWrapper(PretrainedModelsWrapper):

    model_names = ['dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn131', 'dpn107']

    flatten_features_output = False

    def get_original_model(self):
        # The original model is always constructed with test_time_pool=True
        model = getattr(pretrainedmodels, self.model_name)
        if self.pretrained:
            if self.model_name in {'dpn68b', 'dpn92', 'dpn107'}:
                pretrained = 'imagenet+5k'
            else:
                pretrained = 'imagenet'
            model_kwargs = {'pretrained': pretrained, 'num_classes': 1000}
        else:
            model_kwargs = {'pretrained': None}
        return model(**model_kwargs)

    def get_classifier_in_features(self, original_model):
        return original_model.classifier.in_channels

    def get_classifier(self, in_features, num_classes):
        return nn.Conv2d(in_features, num_classes, kernel_size=1, bias=True)

    def classifier(self, x):
        x = self._classifier(x)
        if not self.training:
            x = adaptive_avgmax_pool2d(x, pool_type='avgmax')
        return x.view(x.size(0), -1)


class InceptionV4Wrapper(PretrainedModelsWrapper):

    model_names = ['inception_v4']

    def get_original_model(self):
        if self.pretrained:
            model_kwargs = {'pretrained': 'imagenet', 'num_classes': 1000}
        else:
            model_kwargs = {'pretrained': None}
        return pretrainedmodels.inceptionv4(**model_kwargs)


class XceptionWrapper(PretrainedModelsWrapper):

    model_names = ['xception']

    def get_features(self, original_model):
        return nn.Sequential(*list(original_model.children())[:-1])
