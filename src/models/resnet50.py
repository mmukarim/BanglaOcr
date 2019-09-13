from keras import layers
from keras.layers import Input, Convolution2D, BatchNormalization
from keras.layers import Activation, MaxPooling2D
from keras.models import Model


class ResNet50:
    def __init__(self, num_channels, img_rows, img_cols, num_classes):
        """
        :param num_channels:
        :param img_rows:
        :param img_cols:
        :param num_classes:
        """
        super().__init__()
        self.num_channels = num_channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.num_classes = num_classes

    @staticmethod
    def identity_block(input_tensor, kernel_size, filters, stage, block):
        """
        :param input_tensor:
        :param kernel_size:
        :param filters:
        :param stage:
        :param block:
        :return:
        """
        filters1, filters2, filters3 = filters

        batch_normalization_axis = 3

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Convolution2D(filters1, (1, 1),
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=batch_normalization_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Convolution2D(filters2, kernel_size,
                          padding='same',
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=batch_normalization_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Convolution2D(filters3, (1, 1),
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=batch_normalization_axis, name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = Activation('relu')(x)

        return x

    @staticmethod
    def convolution_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
        """

        :param input_tensor:
        :param kernel_size:
        :param filters:
        :param stage:
        :param block:
        :param strides:
        :return:
        """

        filters1, filters2, filters3 = filters

        batch_normalization_axis = 3

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Convolution2D(filters1, (1, 1), strides=strides,
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=batch_normalization_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Convolution2D(filters2, kernel_size, padding='same',
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=batch_normalization_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Convolution2D(filters3, (1, 1),
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=batch_normalization_axis, name=bn_name_base + '2c')(x)

        shortcut = Convolution2D(filters3, (1, 1), strides=strides,
                                 kernel_initializer='he_normal',
                                 name=conv_name_base + '1')(input_tensor)
        shortcut = layers.BatchNormalization(
            axis=batch_normalization_axis, name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = layers.Activation('relu')(x)

        return x

    def setup(self):
        inputs = Input((self.img_rows, self.img_cols, self.num_channels))
        batch_normalization_axis = 3

        x = Convolution2D(64, (7, 7),
                          strides=(2, 2),
                          padding='valid',
                          kernel_initializer='he_normal',
                          name='conv1')(inputs)
        x = BatchNormalization(axis=batch_normalization_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)

        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = ResNet50.convolution_block(input_tensor=x, kernel_size=3, filters=[64, 64, 256], stage=2, block='a',
                                       strides=(1, 1))
        x = ResNet50.identity_block(input_tensor=x, kernel_size=3, filters=[64, 64, 256], stage=2, block='b')
        x = ResNet50.identity_block(input_tensor=x, kernel_size=3, filters=[64, 64, 256], stage=2, block='c')

        x = ResNet50.convolution_block(input_tensor=x, kernel_size=3, filters=[128, 128, 512], stage=3, block='a')
        x = ResNet50.identity_block(input_tensor=x, kernel_size=3, filters=[128, 128, 512], stage=3, block='b')
        x = ResNet50.identity_block(input_tensor=x, kernel_size=3, filters=[128, 128, 512], stage=3, block='c')
        x = ResNet50.identity_block(input_tensor=x, kernel_size=3, filters=[128, 128, 512], stage=3, block='d')

        x = ResNet50.convolution_block(input_tensor=x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='a')
        x = ResNet50.identity_block(input_tensor=x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='b')
        x = ResNet50.identity_block(input_tensor=x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='c')
        x = ResNet50.identity_block(input_tensor=x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='d')
        x = ResNet50.identity_block(input_tensor=x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='e')
        x = ResNet50.identity_block(input_tensor=x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='f')

        x = ResNet50.convolution_block(input_tensor=x, kernel_size=3, filters=[512, 512, 2048], stage=5, block='a')
        x = ResNet50.identity_block(input_tensor=x, kernel_size=3, filters=[512, 512, 2048], stage=5, block='b')
        x = ResNet50.identity_block(input_tensor=x, kernel_size=3, filters=[512, 512, 2048], stage=5, block='c')

        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(self.num_classes, activation='softmax', name='fc1000')(x)

        model = Model(inputs, x)
        return model

    # def compile(self, optimizer, loss, metrics):
    #     self.compile(optimizer, loss, metrics)
    #
    # def fit_generator(self, train_data, validation_data, steps_per_epoch, epochs, verbose):
    #     self.fit_generator(train_data, validation_data, steps_per_epoch, epochs, verbose)
