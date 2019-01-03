import keras
import cv2
import numpy as np
import os
import random


class DataGenerator(keras.utils.Sequence):
    """Data generator to produce image-mask pairs from the generator array."""
    def __init__(self, X_train, y_train, batch_size=32,
                 img_rows=256, img_cols=256, shuffle=True, horizontal_flip=False,
                 vertical_flip=False, swap_axis=False):
        self.images = X_train
        self.masks = y_train
        self.batch_size = batch_size

        self.img_cols = img_cols
        self.img_rows = img_rows

        self.shuffle = shuffle
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.swap_axis = swap_axis

        self.on_epoch_end()

    def on_epoch_end(self):
        'Update indices, rotations, etc. after each epoch'
        # reorder images
        self.image_indexes = np.arange(self.images.shape[0])
        if self.shuffle:
            np.random.shuffle(self.image_indexes)

    def _data_generation(self,image_idxs):
        # initialize
        X_batch = np.empty((self.batch_size, self.img_rows, self.img_cols, self.images.shape[1]))
        y_batch = np.empty((self.batch_size, self.img_rows, self.img_cols, self.masks.shape[1]))
        X_height = self.images.shape[2]
        X_width = self.masks.shape[3]

        for i in range(self.batch_size):
            random_width = random.randint(0, X_width - self.img_cols - 1)
            random_height = random.randint(0, X_height - self.img_rows - 1)

            random_image = image_idxs[i]

            yb = self.masks[random_image, :, random_height: random_height + self.img_rows,
                         random_width: random_width + self.img_cols]
            xb = np.array(
                self.images[random_image, :, random_height: random_height + self.img_rows, random_width: random_width + self.img_cols])

            if self.horizontal_flip:
                if np.random.random() < 0.5:
                    xb = np.flip(xb, axis=1)
                    yb = np.flip(yb, axis=1)

            if self.vertical_flip:
                if np.random.random() < 0.5:
                    xb = np.flip(xb, axis=2)
                    yb = np.flip(yb, axis=2)
            if self.swap_axis:
                if np.random.random() < 0.5:
                    xb = xb.swapaxes(1, 2)
                    yb = yb.swapaxes(1, 2)


            X_batch[i] = np.rollaxis(xb, 0, 3)
            y_batch[i] = np.rollaxis(yb, 0, 3)

            # print("X batch size:   ", X_batch[i].shape)
            # print("y batch size:   ", y_batch[i].shape)

            # img_pan = X_batch[i][:,:, 0:3]*65536.0
            # np.clip(img_pan, None, 3000, out=img_pan)
            # # finally, rescale to 8-bit range with threshold value scaled to 255
            # img_pan = np.floor_divide(img_pan,
            #                                    3000 / 255).astype('uint8')
            # # mask = imgs_mask[i,0,:,:]
            # ax1 = plt.subplot(121)
            # ax1.set_title("mask")
            # ax1.imshow(y_batch[i][:, :, 0])
            #
            # ax2 = plt.subplot(122)
            # ax2.set_title("img")
            # ax2.imshow(img_pan)
            # plt.show()
        return X_batch,  y_batch[:, 16:16 + self.img_rows - 32, 16:16 + self.img_cols - 32, :]

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.images.shape[1]/self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        im_inds = self.image_indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self._data_generation(image_idxs=im_inds)

        return X, y

