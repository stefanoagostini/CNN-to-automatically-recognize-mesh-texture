# Import
import numpy as np
import cv2
import os


class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        # store the image preprocessors
        self.preprocessors = preprocessors

        # if None then initialize as an empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        # initialize the list of data and labels
        data = []
        labels = []

        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            # preprocessing
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)
            data.append(image)
            labels.append(label)

        # print in console every "verbose" images processed
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] Processed {}/{}".format(i + 1, len(imagePaths)))

        return np.array(data), np.array(labels)
