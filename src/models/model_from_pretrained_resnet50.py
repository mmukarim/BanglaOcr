import keras
from keras.applications import ResNet50
from keras import Model, layers


class PreTrainedResnet50:
    def __init__(self):
        super().__init__()

    def setup(self):
        model_base = ResNet50(include_top=False, weights='imagenet')
        for layer in model_base.layers:
            layer.trainable = False
        x = model_base.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        # x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(1, activation='softmax')(x)

        model = Model(model_base.input, x)
        return model



