from __future__ import division

import os
from tqdm import tqdm
import pandas as pd
import extra_functions
import shapely.geometry
from numba import jit
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
from constants import COLLECTS
from constants import normalization_value
import argparse
from models import h_unet
from keras.models import load_model
from losses import jaccard_coef_loss,jaccard_coef_int
from metrics import precision, recall

def read_model(cross='', weight_path=''):
    weight_name = '8_1000_' + 'Atlanta_nadir53_catid_1030010003CD4300266-0.60_ckpt_best' + '.hdf5'
    print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    print("load model:     ", weight_name)
    print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")

    return load_model(os.path.join(weight_path, weight_name), custom_objects={
        'jaccard_coef_loss': jaccard_coef_loss,
        'jaccard_coef_int': jaccard_coef_int,
        'precision': precision,
        'recall': recall})


# stupid function, should flip directly without swap axis
def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def mask2poly(predicted_mask, threshold):
    polygons = extra_functions.mask2polygons_layer(predicted_mask[0] > threshold, epsilon=1, min_area=5.0)
    return polygons


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test_data', '-td', type=str, required=True,
        help='Path to the directory containing total 27 collects for test.'
    )
    parser.add_argument(
        '--weight_data', '-tc', type=str, required=True,
        help='location of model weight.'
    )
    args = parser.parse_args()

    test_path = args.test_data
    weight_data = args.weight_data

    patch_width = 256
    patch_height = 256
    channels = 13
    threshold = 0.3
    result = []


    model = read_model(weight_path=weight_data)
    for collect_name in [COLLECTS[26]]:
        index = COLLECTS.index(collect_name)

        # model = read_model(cross=collect_name, weight_path=weight_data)
        collect_pansharpen_path = os.path.join(test_path, collect_name, 'Pan-Sharpen')

        print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
        print("read collect:     ", collect_pansharpen_path)
        print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")

        im_list = [f for f in os.listdir(collect_pansharpen_path)
                   if f.endswith('.tif')]
        im_list = ['_'.join(c.rstrip('.tif').split('_')[1:7]) for c in im_list]
        n_ims = len(im_list)
        print(len(im_list), im_list[1])
        print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
        print("normalization_value:     ", normalization_value[index])
        print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")

        for i in range(n_ims):
            image_id = im_list[i]
            print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
            print("predict image id:     ", image_id)
            print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")

            image = extra_functions.read_image_16(test_path, collect_name, image_id, normalization_value[index])

            predicted_mask = extra_functions.make_prediction_cropped(model, image, initial_size=(patch_width, patch_height),
                                                                     final_size=(patch_width-32, patch_height-32),
                                                                     num_masks=1, num_channels=channels)

            image_v = flip_axis(image, 1)
            predicted_mask_v = extra_functions.make_prediction_cropped(model, image_v, initial_size=(patch_width, patch_height),
                                                                       final_size=(patch_width - 32, patch_height - 32),
                                                                       num_masks=1,
                                                                       num_channels=channels)

            image_h = flip_axis(image, 2)
            predicted_mask_h = extra_functions.make_prediction_cropped(model, image_h, initial_size=(patch_width, patch_height),
                                                                       final_size=(patch_width - 32, patch_height - 32),
                                                                       num_masks=1,
                                                                       num_channels=channels)

            image_s = image.swapaxes(1, 2)
            predicted_mask_s = extra_functions.make_prediction_cropped(model, image_s, initial_size=(patch_width, patch_height),
                                                                       final_size=(patch_width - 32, patch_height - 32),
                                                                       num_masks=1,
                                                                       num_channels=channels)

            new_mask = np.power(predicted_mask *
                                flip_axis(predicted_mask_v, 1) *
                                flip_axis(predicted_mask_h, 2) *
                                predicted_mask_s.swapaxes(1, 2), 0.25)

            # ax1 = plt.subplot(111)
            # ax1.set_title(image_id)
            # ax1.imshow(new_mask[0] > threshold)
            # plt.show()

            ix = 0
            polygons = mask2poly(new_mask, threshold)
            for k in range(len(polygons)):
                poly = polygons[k]
                result += [(image_id, ix, str(poly), poly.area)]
                ix = ix + 1

        submission = pd.DataFrame(result, columns=['ImageId', 'BuildingId', 'PolygonWKT_Pix', 'Confidence'])

        #submission.to_csv('ready_for_submission_' + collect_name + 'based_on_42_196_ckpt.csv', index=False)
        submission.to_csv('sample_' + collect_name + 'based_on_53_1_266_ckpt.csv', index=False)

