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


def split_train_test(path_files, path_label):
    """
    Goal: split data into train and test

    :param path_files: data path
    :param path_label: label path

    """
    test = []
    train = []
    for f in glob.glob(path_files + '*'):
        model = os.path.basename(f)
        labels = sio.loadmat(path_label + '/' + model)
        tmp_test = []
        tmp_train = []
        for elem in glob.glob(f + '/*/*.mat'):
            load_elem = sio.loadmat(elem)
            val = load_elem['grid_data'][0][0][0]
            name = int(''.join(filter(str.isdigit, os.path.basename(elem))))
            number = re.findall(r'[0-9]+', str(elem))
            ori = int(number[-1])
            if (labels['label'][val] == 0):
                tmp_test.append((elem, val, model, name, ori, labels['label'][val]))
            else:
                tmp_train.append((elem, val, model, name, ori, labels['label'][val]))
        tmp_train.sort(key=sortFour)
        tmp_test.sort(key=sortFour)
        train = train + tmp_train
        test = test + tmp_test
    return train, test


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
    except OSError:
        print('creation of the directory %s failed', path_folder)
    else:
        ("Succesfully created the directory %s", path_folder)
    return path_folder


def arrayImages(vect, path_f):
    """
    Goal: create an array containing 3 channels characterized by descriptor: localdepth, azimuth,elevation

    :param vect: vector
    :param path_f: path into save image
    """
    for elem in vect:
        mat = sio.loadmat(elem[0])
        facet = elem[1][0][0]
        model = elem[2]
        ori = elem[4]
        label = elem[5][0][0][0]

        localdepth = (mat['grid_data'][0][0][1])*100
        azimuth = (mat['grid_data'][0][0][2])*100
        elevation = (mat['grid_data'][0][0][3])*100

        vect_img = np.array([localdepth, azimuth, elevation])
        vect_img = vect_img.transpose(1, 2, 0)

        facet = np.array2string(facet)  # string
        ori = str(ori)
        label = str(label)
        tmp_path_f = path_f + "/" + label
        saveImages(tmp_path_f + '/' + model + '_' + facet + '_' + ori, vect_img)


def saveImages(path, vect_img):
    """
    Goal: create image end save this

    :param path: path also containing the name of image, into save the image
    :param vect_img: array from which to create the image

    """
    cv2.imwrite(path + '.png', vect_img)


path_files = 'Data/Dataset_grids/'
path_label = 'Data/SHREK18_Labels/'

if __name__ == "__main__":
    path = 'Data/images'
    path_folder_train = createFolder(path, '/train')
    for i in np.arange(1, 13):
        createFolder(path_folder_train, '/' + str(i))
    path_folder_test = createFolder(path, '/test')
    createFolder(path_folder_test, '/0')

    train, test = split_train_test(path_files, path_label)
    arrayImages(train, path_folder_train)
    arrayImages(test, path_folder_test)
