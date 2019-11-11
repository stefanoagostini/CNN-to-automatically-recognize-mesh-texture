# Import
import cv2


class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store width, height and interpolation
        # used when resizing
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # resize an image to a fixed size ignoring aspect ratio
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
