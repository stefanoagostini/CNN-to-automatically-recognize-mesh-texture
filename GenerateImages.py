import scipy.io as sio
import glob
import numpy as np
import os
import cv2
import re


def sortFour(val):
    """
    Goal: orders based on the fourth value

    :param val: vector to order
    :return: return the fourth element of the elements passed as the paramater
    """
    return val[3]


def createFolder(path, name):
    """
    goal:create new folder

    :param path: path into create new folder
    :param name: name of new folder
    :return: new folder path
    """
    path_folder = path + name
    try:
        os.makedirs(path_folder)
    except FileExistsError:
        print(' directory {} already exist'.format(path_folder))
        pass
    except OSError:
        print('creation of the directory {} failed'.format(path_folder))
    else:
        print("Succesfully created the directory {} ".format(path_folder))
    return path_folder


def min_max_pixel(train):
    """
      goal:determinate minimum and maximum of pixels

      :param train: vector containing path with nwe name of images
      :return: values minimum and maximum of total pixels
      """
    for i, elem in enumerate(train):
        mat = sio.loadmat(elem[0])
        if i == 0:
            minimum = np.min(mat['grid_data'][0][0][1])
            maximum = np.max(mat['grid_data'][0][0][1])
        else:
            if minimum > np.min(mat['grid_data'][0][0][1]):
                minimum = np.min(mat['grid_data'][0][0][1])
            if minimum > np.min(mat['grid_data'][0][0][2]):
                minimum = np.min(mat['grid_data'][0][0][2])
            if minimum > np.min(mat['grid_data'][0][0][3]):
                minimum = np.min(mat['grid_data'][0][0][3])
            if maximum < np.max(mat['grid_data'][0][0][1]):
                maximum = np.max(mat['grid_data'][0][0][1])
            if maximum < np.max(mat['grid_data'][0][0][2]):
                maximum = np.max(mat['grid_data'][0][0][2])
            if maximum < np.max(mat['grid_data'][0][0][3]):
                maximum = np.max(mat['grid_data'][0][0][3])
            else:
                pass
    return minimum, maximum


def arrayToImages(path_files, path_label, path_f):
    """
    Goal: create an array containing 3 channels characterized by descriptor: localdepth, azimuth,elevation

    :param vect: vector
    :param path_f: path into save image
    """
    print('[INFO] generate and save images...')
    train = []
    for f in glob.glob(path_files + '*'):
        model = os.path.basename(f)
        labels = sio.loadmat(path_label + '/' + model)
        tmp_train = []
        for elem in glob.glob(f + '/*/*.mat'):
            load_elem = sio.loadmat(elem)
            val = load_elem['grid_data'][0][0][0]
            name = int(''.join(filter(str.isdigit, os.path.basename(elem))))
            number = re.findall(r'[0-9]+', str(elem))
            ori = int(number[-1])
            tmp_train.append((elem, val, model, name, ori, labels['label'][val]))
        tmp_train.sort(key=sortFour)
        train = train + tmp_train

    minimum, maximum = min_max_pixel(train)
    print(minimum, maximum)
    old_range = maximum - minimum
    new_range = 255
    for elem in train:
        mat = sio.loadmat(elem[0])
        facet = elem[1][0][0]
        model = elem[2]
        ori = elem[4]
        label = elem[5][0][0][0]

        scale_l = ((mat['grid_data'][0][0][1]) - minimum) / old_range
        localdepth = (new_range * scale_l)
        scale_a = ((mat['grid_data'][0][0][2]) - minimum) / old_range
        azimuth = (new_range * scale_a)
        scale_e = ((mat['grid_data'][0][0][3]) - minimum) / old_range
        elevation = (new_range * scale_e)

        vect_img = np.array([localdepth, azimuth, elevation])
        vect_img = vect_img.transpose(1, 2, 0)

        facet = np.array2string(facet)
        ori = str(ori)
        label = str(label)
        tmp_path_f = path_f + "/" + label
        saveImages(tmp_path_f + '/' + model + '_' + facet + '_' + ori, vect_img)


def saveImages(path_img, vect_img):
    """
    Goal: create image end save this

    :param path: path also containing the name of image, into save the image
    :param vect_img: array from which to create the image

    """

    cv2.imwrite(path_img + '.png', vect_img)


path_files = 'Data/Dataset_grids/'
path_label = 'Data/SHREK18_Labels/'

if __name__ == "__main__":
    path = 'Data/images'
    for i in np.arange(0, 13):
        createFolder(path, '/' + str(i))

    arrayToImages(path_files, path_label, path)
