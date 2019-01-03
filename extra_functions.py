from shapely.wkt import loads as wkt_loads
import os
import shapely
import shapely.geometry
import shapely.affinity
import h5py
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import csv
import sys

import cv2
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import skimage.color as color
from skimage.transform import rescale

# dirty hacks from SO to allow loading of big cvs's
# without decrement loop it crashes with C error
# http://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
maxInt = sys.maxsize
decrement = True

while decrement:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.
    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True
#
# data_path = '/media/stevehan/data/SpaceNet_Off-Nadir_Dataset/SpaceNet-Off-Nadir_Train'
# train_wkt = pd.read_csv(os.path.join(data_path,'summaryData_Train_2', 'Atlanta_nadir7_catid_1030010003D22F00_Train.csv'))

epsilon = 1e-15



def mask_to_polygons(mask, epsilon=1, min_area=1.):
    # first, find contours with cv2: it's much faster than shapely
    image, contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)

    # create approximate contours to have reasonable submission size
    if epsilon != 0:
        approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                           for cnt in contours]
    else:
        approx_contours = contours

    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            poly = poly.buffer(0)
            all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    # all_polygons = MultiPolygon(all_polygons)
    # if not all_polygons.is_valid:
    #     all_polygons = all_polygons.buffer(0)
    #     # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
    #     # need to keep it a Multi throughout
    #     if all_polygons.type == 'Polygon':
    #         all_polygons = MultiPolygon([all_polygons])
    return all_polygons


def mask2polygons_layer(mask, epsilon=1.0, min_area=10.0):
    # first, find contours with cv2: it's much faster than shapely
    _, contours, hierarchy = cv2.findContours(((mask == 1) * 255).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)

    # create approximate contours to have reasonable submission size
    if epsilon != 0:
        approx_contours = simplify_contours(contours, epsilon)
    else:
        approx_contours = contours

    if not approx_contours:
        return MultiPolygon()

    all_polygons = find_child_parent(hierarchy, approx_contours, min_area)

    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)

    all_polygons = fix_invalid_polygons(all_polygons)

    return all_polygons


def find_child_parent(hierarchy, approx_contours, min_area):
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1

    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])

    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            holes = [c[:, 0, :] for c in cnt_children.get(idx, []) if cv2.contourArea(c) >= min_area]
            contour = cnt[:, 0, :]

            poly = Polygon(shell=contour, holes=holes)

            if poly.area >= min_area:
                all_polygons.append(poly)

    return all_polygons


def simplify_contours(contours, epsilon):
    return [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours]


def fix_invalid_polygons(all_polygons):
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons


def stretch_n(bands, lower_percent=0, higher_percent=100):
    out = np.zeros_like(bands)
    n = bands.shape[2]
    for i in range(n):
        a = 0  # np.min(band)
        b = 1  # np.max(band)
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t

    return out.astype(np.float32)

def stretch_8_n(bands, lower_percent=0, higher_percent=100):
    out = np.zeros_like(bands)
    n = bands.shape[2]
    for i in range(n):
        a = 0  # np.min(band)
        b = 255  # np.max(band)
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t

    return out.astype(np.float32)


def read_image_16(data_path,collect_name, image_id, normalization_value = 65535.0):
    path_MS = os.path.join(data_path, collect_name, 'MS', "MS_{}.tif".format(image_id))
    path_PAN = os.path.join(data_path, collect_name, 'PAN', "PAN_{}.tif".format(image_id))
    path_Pan_Sharpen = os.path.join(data_path, collect_name, 'Pan-Sharpen', "Pan-Sharpen_{}.tif".format(image_id))

    im_reader = rasterio.open(path_MS)
    im_reader_2 = rasterio.open(path_PAN)
    im_reader_3 = rasterio.open(path_Pan_Sharpen)

    img_m = np.empty((im_reader.height,im_reader.width,im_reader.count))
    for band in range(im_reader.count):
        img_m[:, :, band] = im_reader.read(band + 1)

    img_4 = np.empty((im_reader_3.height,im_reader_3.width,im_reader_3.count))
    for band in range(im_reader_3.count):
        img_4[:, :, band] = im_reader_3.read(band + 1)

    img_p = np.empty((im_reader_2.height,im_reader_2.width))
    for band in range(im_reader_2.count):
        img_p[:, :] = im_reader_2.read(band + 1)

    # img_m = np.transpose(tiff.imread(path_MS), (1, 2, 0)) / 65535.0
    # img_4 = np.transpose(tiff.imread(path_Pan_Sharpen), (1, 2, 0)) / 65535.0
    # img_p = tiff.imread(path_PAN).astype(np.float32) /65535.0
    # print("normalization value:   ", normalization_value)

    img_m = img_m / normalization_value
    img_4 = img_4 / normalization_value
    img_p = img_p / normalization_value

    # three_channel_im = img_4[:, :, 0:3]  # remove 4th channel
    # np.clip(three_channel_im, None, 3000, out=three_channel_im)
    # # finally, rescale to 8-bit range with threshold value scaled to 255
    # three_channel_im = np.floor_divide(three_channel_im,
    #                                   3000/255).astype('uint8')
    # ax1 = plt.subplot(111)
    # ax1.set_title(image_id)
    # ax1.imshow(three_channel_im)
    # plt.show()

    # img_m = stretch_n(img_m)
    # img_4 = stretch_n(img_4)
    img_p = np.expand_dims(img_p, 2)
    # img_p = stretch_n(img_p)



    rescaled_M = cv2.resize(img_m, (900, 900), interpolation=cv2.INTER_CUBIC)
    rescaled_M[rescaled_M > 1] = 1
    rescaled_M[rescaled_M < 0] = 0

    # image_r = img_4[:, :, 2]
    # image_g = img_4[:, :, 1]
    # image_b = img_4[:, :, 0]
    # nir = rescaled_M[:, :, 7]
    # re = rescaled_M[:, :, 5]
    #
    # L = 1.0
    # C1 = 6.0
    # C2 = 7.5
    # evi = (nir - image_r) / (nir + C1 * image_r - C2 * image_b + L)
    # evi = np.expand_dims(evi, 2)
    #
    # ndwi = (image_g - nir) / (image_g + nir)
    # ndwi = np.expand_dims(ndwi, 2)
    #
    # savi = (nir - image_r) / (image_r + nir)
    # savi = np.expand_dims(savi, 2)

    result = np.transpose(np.concatenate([rescaled_M, img_p, img_4], axis=2), (2, 0, 1))

    # result = np.transpose(np.concatenate([rescaled_M, img_p, ndwi, savi, evi, img_4], axis=2), (2, 0, 1))
    return result.astype(np.float16)


def make_prediction_cropped(model, X_train, initial_size=(572, 572), final_size=(388, 388), num_channels=3, num_masks=10):
    shift = int((initial_size[0] - final_size[0]) / 2)

    height = X_train.shape[1]
    width = X_train.shape[2]

    if height % final_size[1] == 0:
        num_h_tiles = int(height / final_size[1])
    else:
        num_h_tiles = int(height / final_size[1]) + 1

    if width % final_size[1] == 0:
        num_w_tiles = int(width / final_size[1])
    else:
        num_w_tiles = int(width / final_size[1]) + 1

    rounded_height = num_h_tiles * final_size[0]
    rounded_width = num_w_tiles * final_size[0]

    padded_height = rounded_height + 2 * shift
    padded_width = rounded_width + 2 * shift

    padded = np.zeros((num_channels, padded_height, padded_width))

    padded[:, shift:shift + height, shift: shift + width] = X_train

    # add mirror reflections to the padded areas
    up = padded[:, shift:2 * shift, shift:-shift][:, ::-1]
    padded[:, :shift, shift:-shift] = up

    lag = padded.shape[1] - height - shift
    bottom = padded[:, height + shift - lag:shift + height, shift:-shift][:, ::-1]
    padded[:, height + shift:, shift:-shift] = bottom

    left = padded[:, :, shift:2 * shift][:, :, ::-1]
    padded[:, :, :shift] = left

    lag = padded.shape[2] - width - shift
    right = padded[:, :, width + shift - lag:shift + width][:, :, ::-1]

    padded[:, :, width + shift:] = right

    h_start = range(0, padded_height, final_size[0])[:-1]
    assert len(h_start) == num_h_tiles

    w_start = range(0, padded_width, final_size[0])[:-1]
    assert len(w_start) == num_w_tiles

    temp = []
    for h in h_start:
        for w in w_start:
            temp += [padded[:, h:h + initial_size[0], w:w + initial_size[0]]]
    temp = np.array(temp)
    # print(temp.shape)
    prediction = model.predict(np.rollaxis(temp, 1, 4))

    predicted_mask = np.zeros((rounded_height, rounded_width, num_masks))

    for j_h, h in enumerate(h_start):
         for j_w, w in enumerate(w_start):
             i = len(w_start) * j_h + j_w
             predicted_mask[ h: h + final_size[0], w: w + final_size[0], :] = prediction[i]

    return np.rollaxis(predicted_mask[:height, :width, :], 2, 0)


def generate_mask_for_image(imageId, wkt_list_pandas):
    polygon_list = _get_polygon_list(wkt_list_pandas, imageId)
    contours = _get_and_convert_contours(polygon_list)
    mask = _plot_mask_from_contours(contours)
    return mask


def _plot_mask_from_contours(contours):
    img_mask = np.zeros((900,900), np.uint8)
    if contours is None:
        return img_mask
    perim_list, interior_list = contours
    cv2.fillPoly(img_mask, perim_list, 1)
    cv2.fillPoly(img_mask, interior_list, 0)
    np.set_printoptions(threshold=np.nan)
    return img_mask


def _get_and_convert_contours(polygonList):
    perim_list = []
    interior_list = []
    if polygonList is None:
        return None

    for k in range(len(polygonList)):
        poly = polygonList[k]
        perim = np.array(list(poly.exterior.coords))

        perim_c = _convert_coordinates_to_raster(perim)
        perim_list.append(perim_c)

        for pi in poly.interiors:
            interior = np.array(list(pi.coords))
            interior_c = _convert_coordinates_to_raster(interior)
            interior_list.append(interior_c)
    return perim_list, interior_list


def _convert_coordinates_to_raster(coords):
    coords_int = np.round(coords).astype(np.int32)
    return coords_int

def _get_polygon_list(wkt_list_pandas, imageId):
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
    polygon_list = [wkt_loads(polygon) for polygon in df_image.PolygonWKT_Pix]
    multipolygon = MultiPolygon(polygon_list)
    return multipolygon

