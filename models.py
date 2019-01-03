# random seed instantiated
RANDOM_SEED = 1337
import numpy as np
np.random.seed(RANDOM_SEED)
import tensorflow as tf
tf.set_random_seed(RANDOM_SEED)
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import concatenate, BatchNormalization, Dropout, UpSampling2D,Cropping2D
from keras.models import Model
from keras.optimizers import SGD, Adam, Adagrad, Nadam
from keras.utils import multi_gpu_model
from losses import jaccard_coef_loss,jaccard_coef_int
from metrics import precision, recall
from keras.models import load_model


def compile_model(arch='unet', input_shape=(256, 256, 13), base_depth=64,
                  lr=0.0001, optimizer='Adam', loss_func='binary_crossentropy',
                  additional_metrics=[], verbose=False, **model_args):


    if arch == 'unet':
        print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
        print("Training using valina Unet.")
        print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")

        model = vanilla_unet(input_shape=input_shape, base_depth=base_depth,
                             **model_args)
    elif arch == 'h_unet':
        print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
        print("Training using Han Unet.")
        print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
        model = load_model('cache/8_1000_Atlanta_nadir49_catid_103001000349270091-0.84_ckpt_best.hdf5', custom_objects={
            'jaccard_coef_loss': jaccard_coef_loss,
            'jaccard_coef_int': jaccard_coef_int,
            'precision': precision,
            'recall': recall})
        # origin_model = model.layers[-2]
        #model = h_unet(input_shape=input_shape)
    elif arch == 'ternausnetv1':
        model = ternausnetv1(input_shape=input_shape, base_depth=base_depth)
    else:
        raise ValueError("Unknown model architecture {}".format(arch))

    if optimizer == 'Adam':
        opt_f = Adam(lr=lr)
    elif optimizer == 'SGD':
        opt_f = SGD(lr=lr)
    elif optimizer == 'Adagrad':
        opt_f = Adagrad(lr=lr)
    elif optimizer == 'Nadam':
        opt_f = Nadam(lr=lr)
    else:
        opt_f = optimizer

    try:

        parallel_model = multi_gpu_model(model, cpu_relocation=True,gpus=0)
        print(parallel_model.summary())
        #parallel_model.layers[-2].set_weights(model.get_weights())  # you can check the index of the sequential model with parallel_model.summary()

        print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
        print("Training using multiple GPUs..")
        print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")

    except ValueError:
        parallel_model = model
        print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
        print("Training using single GPU or CPU..")
        print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")

    parallel_model.compile(optimizer=opt_f,
                  loss=loss_func,
                  metrics=['acc', 'mean_squared_error'] + additional_metrics)
    # model.summary()
    return model, parallel_model



def h_unet(input_shape=(256, 256, 13)):
    inputs = Input(input_shape)
    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_uniform')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = keras.layers.ELU()(conv1)

    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_uniform')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = keras.layers.ELU()(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_uniform')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = keras.layers.ELU()(conv2)

    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_uniform')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = keras.layers.ELU()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, padding='same', kernel_initializer='he_uniform')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = keras.layers.ELU()(conv3)

    conv3 = Conv2D(128, 3, padding='same', kernel_initializer='he_uniform')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = keras.layers.ELU()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, padding='same', kernel_initializer='he_uniform')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = keras.layers.ELU()(conv4)

    conv4 = Conv2D(256, 3, padding='same', kernel_initializer='he_uniform')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = keras.layers.ELU()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, 3, padding='same', kernel_initializer='he_uniform')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = keras.layers.ELU()(conv5)

    conv5 = Conv2D(512, 3, padding='same', kernel_initializer='he_uniform')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = keras.layers.ELU()(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4])

    conv6 = Conv2D(256, 3, padding='same', kernel_initializer='he_uniform')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = keras.layers.ELU()(conv6)

    conv6 = Conv2D(256, 3, padding='same', kernel_initializer='he_uniform')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = keras.layers.ELU()(conv6)

    # up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Conv2D(128, 3, padding='same', kernel_initializer='he_uniform')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = keras.layers.ELU()(conv7)
    conv7 = Conv2D(128, 3, padding='same', kernel_initializer='he_uniform')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = keras.layers.ELU()(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Conv2D(64, 3, padding='same', kernel_initializer='he_uniform')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = keras.layers.advanced_activations.ELU()(conv8)
    conv8 = Conv2D(64, 3, padding='same', kernel_initializer='he_uniform')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = keras.layers.ELU()(conv8)

    # up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(32, 3, padding='same', kernel_initializer='he_uniform')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = keras.layers.ELU()(conv9)
    conv9 = Conv2D(32, 3, padding='same', kernel_initializer='he_uniform')(conv9)
    crop9 = Cropping2D(cropping=((16, 16), (16, 16)))(conv9)
    conv9 = BatchNormalization()(crop9)
    conv9 = keras.layers.ELU()(conv9)
    conv10 = Conv2D(1, 1, padding='same',activation='sigmoid')(conv9)
    # conv10 = Convolution2D(num_mask_channels, 1, 1, )(conv9)

    return Model(inputs, conv10)

    
def ternausnetv1(input_shape=(256, 256, 13), base_depth=64):
    """Keras implementation of untrained TernausNet model architecture.

    Arguments:
    ----------
    input_shape (3-tuple): a tuple defining the shape of the input image.
    base_depth (int): the base convolution filter depth for the first layer
        of the model. Must be divisible by two, as the final layer uses
        base_depth/2 filters. The default value, 64, corresponds to the
        original TernausNetV1 depth.

    Returns:
    --------
    An uncompiled Keras Model instance with TernausNetV1 architecture.

    """
    inputs = Input(input_shape)
    conv1 = Conv2D(base_depth, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2_1 = Conv2D(base_depth*2, 3, activation='relu',
                     padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_1)

    conv3_1 = Conv2D(base_depth*4, 3, activation='relu',
                     padding='same')(pool2)
    conv3_2 = Conv2D(base_depth*4, 3, activation='relu',
                     padding='same')(conv3_1)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_2)

    conv4_1 = Conv2D(base_depth*8, 3, activation='relu',
                     padding='same')(pool3)
    conv4_2 = Conv2D(base_depth*8, 3, activation='relu',
                     padding='same')(conv4_1)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4_2)

    conv5_1 = Conv2D(base_depth*8, 3, activation='relu',
                     padding='same')(pool4)
    conv5_2 = Conv2D(base_depth*8, 3, activation='relu',
                     padding='same')(conv5_1)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5_2)

    conv6_1 = Conv2D(base_depth*8, 3, activation='relu',
                     padding='same')(pool5)

    up7 = Conv2DTranspose(base_depth*4, 2, strides=(2, 2), activation='relu',
                          padding='same')(conv6_1)
    concat7 = concatenate([up7, conv5_2])
    conv7_1 = Conv2D(base_depth*8, 3, activation='relu',
                     padding='same')(concat7)

    up8 = Conv2DTranspose(base_depth*4, 2, strides=(2, 2), activation='relu',
                          padding='same')(conv7_1)
    concat8 = concatenate([up8, conv4_2])
    conv8_1 = Conv2D(base_depth*8, 3, activation='relu',
                     padding='same')(concat8)

    up9 = Conv2DTranspose(base_depth*2, 2, strides=(2, 2), activation='relu',
                          padding='same')(conv8_1)
    concat9 = concatenate([up9, conv3_2])
    conv9_1 = Conv2D(base_depth*4, 3, activation='relu',
                     padding='same')(concat9)

    up10 = Conv2DTranspose(base_depth, 2, strides=(2, 2), activation='relu',
                           padding='same')(conv9_1)
    concat10 = concatenate([up10, conv2_1])
    conv10_1 = Conv2D(base_depth*2, 3, activation='relu',
                      padding='same')(concat10)

    up11 = Conv2DTranspose(int(base_depth/2), 2, strides=(2, 2),
                           activation='relu', padding='same')(conv10_1)
    concat11 = concatenate([up11, conv1])

    out = Conv2D(1, 1, activation='sigmoid', padding='same')(concat11)

    return Model(input=inputs, output=out)


def vanilla_unet(input_shape=(256, 256, 13), base_depth=32, drop_rate=0,
                 seed=1337):
    """Keras vanilla unet architecture implementation.

    Arguments:
    ----------
    input_shape (3-tuple): a tuple defining the shape of the input image.
    base_depth (int): the base convolution filter depth for the first layer
        of the model. Must be divisible by two, as the final layer uses
        base_depth/2 filters. The default value, 64, corresponds to the
        original TernausNetV1 depth.
    drop_rate (int): Dropout rate for dropout layers. Defaults to no dropout.
    seed (int): Random seed for dropout node selection. Defaults to 1337
        because we're nerds.

    Returns:
    --------
    An uncompiled Keras model with vanilla UNet architecture.

    """
    input = Input(input_shape)

    conv1 = Conv2D(base_depth, 3, activation='relu', padding='same')(input)
    bn1 = BatchNormalization()(conv1)
    drop1 = Dropout(drop_rate, seed=seed)(bn1)
    conv2 = Conv2D(base_depth, 3, activation='relu', padding='same')(drop1)
    bn2 = BatchNormalization()(conv2)
    mp1 = MaxPooling2D(pool_size=(2, 2))(bn2)

    conv3 = Conv2D(base_depth*2, 3, activation='relu', padding='same')(mp1)
    bn3 = BatchNormalization()(conv3)
    drop2 = Dropout(drop_rate, seed=seed+1)(bn3)
    conv4 = Conv2D(base_depth*2, 3, activation='relu', padding='same')(drop2)
    bn4 = BatchNormalization()(conv4)
    mp2 = MaxPooling2D(pool_size=(2, 2))(bn4)

    conv5 = Conv2D(base_depth*4, 3, activation='relu', padding='same')(mp2)
    bn5 = BatchNormalization()(conv5)
    drop3 = Dropout(drop_rate, seed=seed+2)(bn5)
    conv6 = Conv2D(base_depth*4, 3, activation='relu', padding='same')(drop3)
    bn6 = BatchNormalization()(conv6)
    mp3 = MaxPooling2D(pool_size=(2, 2))(bn6)

    conv7 = Conv2D(base_depth*8, 3, activation='relu', padding='same')(mp3)
    bn7 = BatchNormalization()(conv7)
    drop4 = Dropout(drop_rate, seed=seed+3)(bn7)
    conv8 = Conv2D(base_depth*8, 3, activation='relu', padding='same')(drop4)
    bn8 = BatchNormalization()(conv8)
    mp4 = MaxPooling2D(pool_size=(2, 2))(bn8)

    conv9 = Conv2D(base_depth*16, 3, activation='relu', padding='same')(mp4)
    bn9 = BatchNormalization()(conv9)
    drop5 = Dropout(drop_rate, seed=seed+4)(bn9)
    deconv0 = Conv2DTranspose(base_depth*16, 3, activation='relu',
                              padding='same')(drop5)
    bn10 = BatchNormalization()(deconv0)
    up1 = UpSampling2D(interpolation='bilinear')(bn10)

    deconv1 = Conv2DTranspose(base_depth*8, 3, activation='relu',
                              padding='same')(up1)
    bn11 = BatchNormalization()(deconv1)
    cat1 = concatenate([bn11, bn8])
    drop6 = Dropout(drop_rate, seed=seed+5)(cat1)
    deconv2 = Conv2DTranspose(base_depth*8, 3, activation='relu',
                              padding='same')(drop6)
    bn12 = BatchNormalization()(deconv2)
    up2 = UpSampling2D(interpolation='bilinear')(bn12)

    deconv3 = Conv2DTranspose(base_depth*4, 3, activation='relu',
                              padding='same')(up2)
    bn13 = BatchNormalization()(deconv3)
    cat2 = concatenate([bn13, bn6])
    drop7 = Dropout(drop_rate, seed=seed+6)(cat2)
    deconv4 = Conv2DTranspose(base_depth*4, 3, activation='relu',
                              padding='same')(drop7)
    bn14 = BatchNormalization()(deconv4)
    up3 = UpSampling2D(interpolation='bilinear')(bn14)

    deconv5 = Conv2DTranspose(base_depth*2, 3, activation='relu',
                              padding='same')(up3)
    bn15 = BatchNormalization()(deconv5)
    cat3 = concatenate([bn15, bn4])
    drop8 = Dropout(drop_rate, seed=seed+7)(cat3)
    deconv6 = Conv2DTranspose(base_depth*2, 3, activation='relu',
                              padding='same')(drop8)
    bn16 = BatchNormalization()(deconv6)
    up4 = UpSampling2D(interpolation='bilinear')(bn16)

    deconv7 = Conv2DTranspose(base_depth, 3, activation='relu',
                              padding='same')(up4)
    bn17 = BatchNormalization()(deconv7)
    cat4 = concatenate([bn17, bn2])
    drop7 = Dropout(drop_rate, seed=seed+8)(cat4)
    deconv8 = Conv2DTranspose(base_depth, 3, activation='relu',
                              padding='same')(drop7)
    bn18 = BatchNormalization()(deconv8)

    out = Conv2DTranspose(1, 1, activation='sigmoid', padding='same')(bn18)
    return Model(input, out)
