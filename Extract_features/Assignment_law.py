import numpy as np
import cv2
from random import randint, randrange
from scipy import signal as sg
from pandas import DataFrame
import pandas as pd


def getTemfeatures(a,b):
    image = cv2.imread('train/%s/%s%d.jpg' % (a,a,b))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (rows, cols) = gray.shape[:2]

    #image pre_processing
    smooth_kernel = (1/25)*np.ones((5,5))
    gray_smooth = sg.convolve(gray, smooth_kernel, "same")
    gray_processed = np.abs(gray - gray_smooth)

    filter_vectors = np.array([[1, 4, 6, 4, 1],
                               [-1, -2, 0, 2, 1],
                               [-1, 0, 2, 0, 1],
                               [1, -4, 6, -4, 1]])

    # patch size
    PATCH_SIZE = 16

    #make filter
    filters = list()
    for i in range(4):
        for j in range(4):
            filters.append(np.matmul(filter_vectors[i][:].reshape(5, 1),
                                     filter_vectors[j][:].reshape(1, 5)))

    # convolution & convmap
    TEM_LIST = list()
    for try_num in range(15):
        # === Convolution 연산 및 convmap 조합 === #
        rand_x = randint(0, image.shape[0] - 1)
        rand_y = randint(0, image.shape[1] - 1)
        croped_image = gray_processed[rand_x:rand_x + PATCH_SIZE, rand_y:rand_y + PATCH_SIZE]
        (rows, cols) = croped_image.shape[:2]

        conv_maps = np.zeros((rows, cols, 16))
        for i in range(len(filters)):
            conv_maps[:, :, i] = sg.convolve(croped_image, filters[i], 'same')

        # texture map calculation === #
        texture_maps = list()
        texture_maps.append((conv_maps[:, :, 1] + conv_maps[:, :, 4]) // 2)  # L5E5 / E5L5
        texture_maps.append((conv_maps[:, :, 2] + conv_maps[:, :, 8]) // 2)  # L5S5 / S5L5
        texture_maps.append((conv_maps[:, :, 3] + conv_maps[:, :, 12]) // 2)  # L5R5 / R5L5
        texture_maps.append((conv_maps[:, :, 7] + conv_maps[:, :, 13]) // 2)  # E5R5 / R5E5
        texture_maps.append((conv_maps[:, :, 6] + conv_maps[:, :, 9]) // 2)  # E5S5 / S5E5
        texture_maps.append((conv_maps[:, :, 11] + conv_maps[:, :, 14]) // 2)  # S5R5 / R5S5
        texture_maps.append((conv_maps[:, :, 10]))  # S5S5
        texture_maps.append((conv_maps[:, :, 5]))  # E5E5
        texture_maps.append((conv_maps[:, :, 15]))  # R5R5
        texture_maps.append((conv_maps[:, :, 0]))  # L5L5

        # === Law's texture energy calculation ===
        TEM = list()
        for i in range(9):
            TEM.append(np.abs(texture_maps[i]).sum() /
                       np.abs(texture_maps[9]).sum())

        TEM_LIST.append(TEM)

    return TEM_LIST

if __name__ == "__main__":
    classname = ['brick', 'grass', 'ground', 'water', 'wood']
    
    df = pd.DataFrame()

    df.to_csv("brick_law.csv", index=False)
    df.to_csv("grass_law.csv", index=False)
    df.to_csv("ground_law.csv", index=False)
    df.to_csv("water_law.csv", index=False)
    df.to_csv("wood_law.csv", index=False)

    for _, str in enumerate(classname):
        result_lst = list()
        for idx in range(1, 11):
            result_lst = getTemfeatures(str, idx)
            if str == 'brick':
                df = pd.DataFrame(result_lst)
                df.to_csv("brick_law.csv", mode='a', index=False, header=False)
            if str == 'brick':
                df = pd.DataFrame(result_lst)
                df.to_csv("grass_law.csv", mode='a', index=False, header=False)
            if str == 'brick':
                df = pd.DataFrame(result_lst)
                df.to_csv("ground_law.csv", mode='a', index=False, header=False)
            if str == 'brick':
                df = pd.DataFrame(result_lst)
                df.to_csv("water_law.csv", mode='a', index=False, header=False)
            if str == 'brick':
                df = pd.DataFrame(result_lst)
                df.to_csv("wood_law.csv", mode='a', index=False, header=False)
