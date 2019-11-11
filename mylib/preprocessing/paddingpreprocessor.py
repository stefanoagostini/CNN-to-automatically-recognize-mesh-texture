# Import
import cv2


class PaddingPreprocessor:
    def __init__(self, paddingTop, paddingBottom, paddingLeft, paddingRight, borderValue):
        # store padding to
        # use when preprocessing
        self.paddingTop = paddingTop
        self.paddingBottom = paddingBottom
        self.paddingLeft = paddingLeft
        self.paddingRight = paddingRight
        self.borderValue = borderValue

    def preprocess(self, image):
        # apply padding to the image
        return cv2.copyMakeBorder(image, self.paddingTop, self.paddingBottom, self.paddingLeft, self.paddingRight, borderType=cv2.BORDER_CONSTANT, value=self.borderValue)