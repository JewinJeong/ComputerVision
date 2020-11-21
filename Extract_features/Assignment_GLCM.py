import cv2
from random import randint, randrange
from skimage.feature import greycomatrix, greycoprops
from pandas import DataFrame

def getFeatures(a, b):
    #read image
    image = cv2.imread('train/%s/%s%d.jpg' %(a,a,b), cv2.IMREAD_GRAYSCALE)

    #patch size
    PATCH_SIZE = 16

    #random box
    brick_locations = list()
    for i in range(100):
        brick_locations.append((randrange(image.shape[0]-16), randrange(image.shape[1]-16)))

    brick_patches = list()
    for loc in brick_locations:
        brick_patches.append(image[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE])

    #glcm feature list
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

    #insert dataframe
    data = {'dissimilarity': dis,
            'correlation': correlation,
            'contrast': contrast,
            'homogeneity': homogeneity,
            'energy': energy,
            'ASM': asm
            }
    frame = DataFrame(data)

    #insert to csv file
    if a == 'brick':
        frame.to_csv('result_brick.csv',mode='a', index=False)
    elif a == 'grass':
        frame.to_csv('result_grass.csv',mode='a', index=False)
    elif a == 'ground':
        frame.to_csv('result_ground.csv',mode='a', index=False)
    elif a == 'water':
        frame.to_csv('result_water.csv',mode='a', index=False)
    elif a == 'wood':
        frame.to_csv('result_wood.csv',mode='a', index=False)
    print(frame)

if __name__ == "__main__":
    #이미지 읽기
    classname = ['brick', 'grass', 'ground', 'water', 'wood']

    for _, str in enumerate(classname):
        for idx in range(1, 11):
            getFeatures(str, idx)
