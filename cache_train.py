"""
Script that caches train data for future training
"""
import os
import pandas as pd
import extra_functions
from tqdm import tqdm
import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt
from constants import COLLECTS
from constants import normalization_value


def cache_train_16():
    num_channels = 13
    num_mask_channels = 1
    for label_csv in tqdm(sorted(os.listdir(label_path))):
        # collect: type int, choose the COLLECT to transform
        if not COLLECTS[collect] in label_csv:
            continue
        train_wkt = pd.read_csv(os.path.join(image_path, 'summaryData_Train_2', label_csv))

        print("image number in collect :  ", label_csv,'   ', train_wkt['ImageId'].nunique(), label_csv.rstrip('.csv'))
        num_train = train_wkt['ImageId'].nunique()

        f = h5py.File(os.path.join(output_train_path, label_csv.rstrip('.csv') + '.h5'), 'w')
        imgs = f.create_dataset('train', (num_train, num_channels, 900, 900), dtype=np.float16)
        imgs_mask = f.create_dataset('train_mask', (num_train, num_mask_channels, 900, 900), dtype=np.uint8)

        i = 0
        print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
        print("normalization_value:     ", normalization_value[collect])
        print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")

        for image_id in tqdm(sorted(train_wkt['ImageId'].unique())):
            # if image_id in imperfect_images:
            #     print("imperfect pictures:   ", image_id)
            #     continue

            imgs[i] = extra_functions.read_image_16(image_path, label_csv.rstrip('_Train.csv'), image_id,
                                                    normalization_value[collect])
            imgs_mask[i] = extra_functions.generate_mask_for_image(image_id, train_wkt)

            # mask = imgs_mask[i, 0, :, :]
            # print(mask.shape)
            # ax1 = plt.subplot(111)
            # ax1.set_title(image_id)
            # ax1.imshow(mask)
            # plt.show()
            i = i + 1

        f.close()


if __name__ == '__main__':
    # image_path = '/media/stevehan/data/SpaceNet_Off-Nadir_Dataset/SpaceNet-Off-Nadir_Train'
    #
    # output_train_path = '/media/stevehan/data/SpaceNet_Off-Nadir_Dataset/train_h5/'
    # train_wkt = pd.read_csv(os.path.join(data_path,'summaryData_Train_2', 'Atlanta_nadir7_catid_1030010003D22F00_Train.csv'))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_data', '-td', type=str, required=True,
        help='Path to the directory containing total 27 collects for train.'
    )
    parser.add_argument(
        '--output_train_path', '-output', type=str, required=True,
        help='choose output path for h5 files.'
    )
    parser.add_argument(
        '--train_collect', '-tc', type=int, required=True,
        help='choose which collect to transform.'
    )
    args = parser.parse_args()

    image_path = args.train_data
    output_train_path = args.output_train_path
    collect = args.train_collect

    if not os.path.exists(output_train_path):
        os.makedirs(output_train_path)

    label_path = os.path.join(image_path, 'summaryData_Train_2')

    cache_train_16()

