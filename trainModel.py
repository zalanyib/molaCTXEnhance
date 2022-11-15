import numpy as np
from libtiff import TIFF
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, Conv2DTranspose, UpSampling2D, Concatenate, Add, Multiply, ReLU, LeakyReLU, PReLU, BatchNormalization, Dropout
import random
from PIL import Image
from pathlib import Path

def createEnhanceModel(dim):
    initializer = tf.random_normal_initializer(0., 0.03)
    inputs = Input(shape=(2*dim, 2*dim, 1)) # 32

    size = 4
    base = 96

    # Downsampling
    l0 = Sequential()
    l0.add(Conv2D(base, size, strides=2, padding='same', use_bias=True, kernel_initializer=initializer))
    l0.add(LeakyReLU()) # 16
    l0 = l0(inputs)

    l1 = Sequential()
    l1.add(Conv2D(base*2, size, strides=2, padding='same', use_bias=False, kernel_initializer=initializer))
    l1.add(BatchNormalization())
    l1.add(LeakyReLU()) # 8
    l1 = l1(l0)

    l2 = Sequential()
    l2.add(Conv2D(base*4, size, strides=2, padding='same', use_bias=False, kernel_initializer=initializer))
    l2.add(BatchNormalization())
    l2.add(LeakyReLU())
    l2 = l2(l1) # 4

    l3 = Sequential()
    l3.add(Conv2D(base*4, size, strides=2, padding='same', use_bias=False, kernel_initializer=initializer))
    l3.add(BatchNormalization())
    l3.add(LeakyReLU()) # 2
    l3 = l3(l2)

    l4 = Sequential()
    l4.add(Conv2D(base*4, size, strides=2, padding='same', use_bias=False, kernel_initializer=initializer))
    l4.add(BatchNormalization())
    l4.add(LeakyReLU()) # 2
    l4 = l4(l3)

    # Upsampling
    l4_2 = Sequential()
    l4_2.add(UpSampling2D(size=2, interpolation='bilinear'))
    l4_2.add(Conv2DTranspose(base*4, size, strides=1, padding='same', use_bias=False, kernel_initializer=initializer))
    l4_2.add(BatchNormalization())
    l4_2.add(Dropout(0.36))
    l4_2.add(ReLU()) # 4
    l4_2 = l4_2(l4)
    l4_2 = Concatenate()([l4_2, l3])

    l5 = Sequential()
    l5.add(UpSampling2D(size=2, interpolation='bilinear'))
    l5.add(Conv2DTranspose(base*4, size, strides=1, padding='same', use_bias=False, kernel_initializer=initializer))
    l5.add(BatchNormalization())
    l5.add(Dropout(0.18))
    l5.add(ReLU()) # 4
    l5 = l5(l4_2)
    l5 = Concatenate()([l5, l2])
    l6 = Sequential()
    l6.add(UpSampling2D(size=2, interpolation='bilinear'))
    l6.add(Conv2DTranspose(base*2, size, strides=1, padding='same', use_bias=False, kernel_initializer=initializer))
    l6.add(BatchNormalization())
    l6.add(ReLU()) # 8
    l6 = l6(l5)
    l6 = Concatenate()([l6, l1])
    l7 = Sequential()
    l7.add(UpSampling2D(size=2, interpolation='bilinear'))
    l7.add(Conv2DTranspose(base, size, strides=1, padding='same', use_bias=False, kernel_initializer=initializer))
    l7.add(BatchNormalization())
    l7.add(ReLU()) # 16
    l7 = l7(l6)
    l7 = Concatenate()([l7, l0])

    out = UpSampling2D(size=2, interpolation='bilinear')(l7)
    out = Conv2DTranspose(1, size, strides=1, padding='same', activation='tanh', kernel_initializer=initializer)(out)

    model = Model(inputs=inputs, outputs=out)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0008, beta_1=0.89)
    model.compile(loss='mean_squared_error', optimizer=opt)
    print(model.summary())
    return model

def train(model, directory, dim, suffix, outputDirectory):
    numItems = 0
    for p in Path(directory).glob('*_x.tif'):
        numItems += 1
    print('[ LOG ] Loading {} training items from {}'.format(numItems, directory))
    X1 = np.zeros((numItems, 2*dim, 2*dim, 1))
    y = np.zeros((numItems, 2*dim, 2*dim, 1))
    indices = np.zeros(numItems, dtype=int)
    for i in range(0, numItems):
        indices[i] = i
    random.shuffle(indices)
    i = 0
    for p in Path(directory).glob('*_x.tif'):
        xName = p.name
        idx = indices[i]
        X1[idx,:,:,0] = TIFF.open(directory+xName).read_image()
        yName = xName.replace('_x.tif', '_y.tif')
        y[idx,:,:,0] = TIFF.open(directory+yName).read_image()
        i += 1

    model.fit(X1, y, epochs=50, batch_size=64, validation_split=0.27)
    
    print("[ LOG ] Evaluating model")
    test = model.evaluate(X1[-640:], y[-640:], batch_size=64)
    if test<1e-3:
        print("[ LOG ] Enhancer probably okay.")
    else:
        print("[ LOG ] Enhancer loss above 1e-3, possibly not good enough.")
    # Save
    model.save(outputDirectory)

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    dim = 32
    suffix = 'phase2'
    trainingDirectory = 'f:/programming/resources/mars/DEM Enhancement/trainingdata_{}/'.format(suffix)
    outputDirectory = 'f:/programming/resources/mars/DEM Enhancement/models/model_enhance_{}'.format(suffix)
    model = createEnhanceModel(dim)
    train(model, trainingDirectory, dim, suffix)
