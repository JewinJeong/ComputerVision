import cv2
from random import randint
from skimage.feature import greycomatrix, greycoprops
from pandas import DataFrame

image = cv2.imread('C:/Users/jewin/PycharmProjects/GLCMTest/train/brick/brick1.jpg', cv2.IMREAD_GRAYSCALE)

PATCH_SIZE = 16
brick_locations = list()
# weigth = image.shape[1] - PATCH_SIZE
# height = image.shape[0] - PATCH_SIZE

for i in range(100):
    brick_locations.append((randint(1, 434), randint(1, 434)))

brick_patches = list()

for loc in brick_locations:
   brick_patches.append(image[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE])

dis = list()
correlation = list()
contrast = list()
homogeneity = list()
energy = list()
asm = list()

for patch in brick_patches:
    glcm = greycomatrix(patch, distances=[1], angles=[0], levels=256, symmetric=False, normed=True)
    dis.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    correlation.append(greycoprops(glcm, 'correlation')[0, 0])
    contrast.append(greycoprops(glcm, 'contrast')[0, 0])
    homogeneity.append(greycoprops(glcm, 'homogeneity')[0, 0])
    energy.append(greycoprops(glcm, 'energy')[0, 0])
    asm.append(greycoprops(glcm, 'ASM')[0, 0])

data = {'dissimilarity': dis,
        'correlation': correlation,
        'contrast': contrast,
        'homogeneity': homogeneity,
        'energy': energy,
        'ASM': asm
        }

frame = DataFrame(data)
print(frame)