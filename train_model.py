import os
import argparse
import warnings
import numpy as np
import h5py
import datetime
# set random seed for numpy and tensorflow
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from DataGenerator import DataGenerator
from callback import TerminateOnMetricNaN
from losses import jaccard_coef_loss,jaccard_coef_int
from metrics import precision, recall
from models import compile_model
from constants import COLLECTS
from keras.utils.vis_utils import plot_model
from keras.models import load_model
import keras


# class MyCbk(keras.callbacks.Callback):
#
#     def __init__(self, model):
#         self.model_to_save = model
#
#     def on_epoch_end(self, epoch, logs=None):
#         self.model_to_save.save('model_at_epoch_%d.h5' % epoch)
#


class MyCbk(keras.callbacks.Callback):
    def __init__(self, model, filepath):
        self.model_to_save = model
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        filepath = self.filepath.format(epoch=epoch + 1, **logs)
        # self.model_to_save.save('test_model_at_epoch_%d.h5' % epoch)

        self.model_to_save.save(filepath, overwrite=True)

def main(train_x, train_y, patch_rows, patch_cols, batch_size=64, epochs=1000,  model="h_unet",name_suffix="suffix",):
    # create a few variables needed later.
    tmp_model_path = os.path.join('cache', name_suffix + '{epoch:02d}-{val_loss:.2f}_ckpt_best.hdf5')

    early_stopping_patience = 15
    model_args = {
        'optimizer': 'Nadam',
        'input_shape': (patch_rows, patch_cols, 13),
        'base_depth': 64,
        'lr': 0.0001
    }
    # reduce base_depth to 32 if using vanilla unet
    if model == 'unet':
        model_args['base_depth'] = 32

    # load in data. don't read entirely into memory - too big.
    # train_im_arr = train_x
    # val_im_arr = train_x
    # train_mask_arr = train_y
    # val_mask_arr = train_y
    # (1064, 13, 900, 900)
    print(train_x.shape)
    print(train_y.shape)
    # create generators for training and validation
    training_gen = DataGenerator(
        train_x, train_y, batch_size=batch_size, img_rows=patch_rows, img_cols=patch_cols,
        horizontal_flip=True, vertical_flip=True, swap_axis=True
        )
    validation_gen = DataGenerator(
        train_x[0:10, :, :, :], train_y[0:10, :, :, :], batch_size=batch_size,
        )
    n_train_ims = train_x.shape[0]
    n_val_ims = 10

    monitor = 'val_loss'
    print()
    print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    print("                 BEGINNING MODEL TRAINING")
    print("                 MODEL ARCHITECTURE: {}".format(model))
    print("                   OPTIMIZER: {}".format(model_args['optimizer']))
    print("                 INPUT SHAPE: {}".format(model_args['input_shape']))
    print("                      BATCH SIZE: {}".format(batch_size))
    print("                   LEARNING RATE: {}".format(model_args['lr']))
    print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    print()

    callbax = []


    # callbax.append(ReduceLROnPlateau(factor=0.2, patience=3, verbose=1,
    #                                  min_delta=0.01))
    # callbax.append(ModelCheckpoint(tmp_model_path, monitor=monitor))
    # callbax.append(TerminateOnMetricNaN('precision'))
    # callbax.append(EarlyStopping(monitor=monitor,
    #                              patience=early_stopping_patience,
    #                              mode='auto'))
    callbax.append(TensorBoard(log_dir='tensorboard'))

    lf = jaccard_coef_loss
    am = [precision, recall, jaccard_coef_int]

    original_model, parallel_model = compile_model(arch=model, loss_func=lf,
                          additional_metrics=am,
                          verbose=True, **model_args)

    callbax.append(MyCbk(original_model, tmp_model_path))
    # plot_model(model, to_file='scheme.png', show_shapes=True)
    # model = load_model('cache/8_1000_Atlanta_nadir44_catid_1030010003CCD70076-0.47_ckpt_best.hdf5', custom_objects={
    #     'jaccard_coef_loss': jaccard_coef_loss,
    #     'jaccard_coef_int': jaccard_coef_int,
    #     'precision': precision,
    #     'recall': recall})

    parallel_model.fit_generator(training_gen, validation_data=validation_gen, validation_steps=np.floor(n_val_ims/batch_size),
                        steps_per_epoch=400,
                        epochs=epochs,workers=8, callbacks=callbax)

    original_model.save(os.path.join('cache', 'model' + name_suffix + '.hdf5'))
    print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    print("                   MODEL TRAINING COMPLETE!                 ")
    print("   Model located at {}".format(name_suffix))
    print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_data', '-td', type=str, required=True,
        help='Path to the directory containing total 27 h5 files for train.'
    )
    parser.add_argument(
        '--train_collect', '-tc', type=int, required=True,
        help='choose which collect to train.'
    )
    args = parser.parse_args()

    # data_path = '/media/stevehan/data/SpaceNet_Off-Nadir_Dataset/SpaceNet-Off-Nadir_Train'
    data_path = args.train_data
    train_collect = args.train_collect
    if not COLLECTS[train_collect] + '_Train.h5' in os.listdir(data_path):
        print("error, check  please!")

    if train_collect < 8:
        img_cols = 256
        img_rows = 256

    if train_collect >= 8 and train_collect <=18:
        img_cols = 256
        img_rows = 256

    if train_collect > 18:
        img_rows = 256
        img_cols = 256

    print("input image size: ", img_cols, img_rows)
    now = datetime.datetime.now()

    print('[{}] Reading train...'.format(str(datetime.datetime.now())))
    print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    print("                   Reading train!                 ")
    print(data_path, '  ', COLLECTS[train_collect], '_Train.h5')
    print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    f = h5py.File(os.path.join(data_path, COLLECTS[train_collect] + '_Train.h5'), 'r')

    X_train = f['train']
    y_train = np.array(f['train_mask'])[:, 0]
    y_train = np.expand_dims(y_train, 1)

    print(X_train[0][8])
    batch_size = 8

    nb_epoch = 1000

    suffix = COLLECTS[train_collect]
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    if not os.path.isdir('tensorboard'):
        os.mkdir('tensorboard')
    main(X_train, y_train, img_rows, img_cols, batch_size, nb_epoch, model="h_unet",
         name_suffix='{batch}_{epoch}_{suffix}'.format(batch=batch_size, epoch=nb_epoch, suffix=suffix))
