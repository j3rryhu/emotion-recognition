from keras import models, layers


def make_block(stage, step, channel, kernel_size=(3,3), stride=(1,1), padding='valid', activation=True):
    block = []
    block.append(layers.Conv2D(channel, kernel_size, stride, use_bias=False, padding=padding))
    block.append(layers.BatchNormalization(name='block{}_conv{}_bn'.format(stage, step)))
    if activation:
        block.append(layers.Activation('selu', name='block{}_conv{}_act'.format(stage, step)))
    return block


def cnn(input_shape, num_classes):
    model = models.Sequential()
    model.add([layers.Convolution2D(filters=16, kernel_size=(7, 7), padding='same', name='image_array', input_shape=input_shape),
               layers.BatchNormalization(name='block1_conv1_bn')])
    block1_2 = make_block(1, 2, 16, kernel_size=(7,7))
    model.add(block1_2)
    model.add(layers.AveragePooling2D(pool_size=(2,2), padding='same'))
    model.add(layers.Dropout(.5))
    block2_1 = make_block(2, 1, 32, kernel_size=(5,5), padding='same', activation=False)
    model.add(block2_1)
    block2_2 = make_block(2, 2, 32, kernel_size=(5,5), padding='same')
    model.add(block2_2)
    model.add(layers.AveragePooling2D(pool_size=(2,2), padding='same'))
    model.add(layers.Dropout(.5))
    block3_1 = make_block(3, 1, 64, padding='same')
    model.add(block3_1)
    block3_2 = make_block(3, 2, 64, padding='same')
    model.add(block3_2)
    model.add(layers.AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(layers.Dropout(.5))
    block4_1 = make_block(4, 1, 128, padding='same')
    model.add(block4_1)
    block4_2 = make_block(4, 2, 128, padding='same')
    model.add(block4_2)
    model.add(layers.AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(layers.Dropout(.5))
    block5_1 = make_block(5, 1, 256, padding='same', activation=False)
    model.add(block5_1)
    model.add(layers.Convolution2D(filters=num_classes, kernel_size=(3, 3), padding='same'),
              layers.GlobalAveragePooling2D(),
              layers.Activation('softmax', name='predictions'))
    return model



