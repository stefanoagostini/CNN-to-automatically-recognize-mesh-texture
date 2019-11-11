from keras.models import load_model
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from mylib.classification_report import classification_report_csv
from mylib.datasets import SimpleDatasetLoader
from mylib.preprocessing import SimplePreprocessor
from mylib.preprocessing import PaddingPreprocessor
from imutils import paths
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np

if __name__ == '__main__':
    print("[INFO] Loading Images...")
    imagePaths = paths.list_images("Data/images/train")
    sp = SimplePreprocessor(64, 64)
    pp = PaddingPreprocessor(32, 32, 32, 32, borderValue=0)
    sdl = SimpleDatasetLoader(preprocessors=[sp, pp])
    (train_X, train_Y) = sdl.load(imagePaths)
    le = preprocessing.LabelEncoder()
    train_Y = le.fit_transform(train_Y)
    train_Y = to_categorical(train_Y)

    x_train, x_valid, y_train, y_valid = train_test_split(train_X, train_Y, test_size=0.2, shuffle=True, random_state=50)
    print("[INFO] predicting...")
    model = load_model('Data/Model/model.hdf5')
    predictions = model.predict(x_valid, batch_size=32)
    predicted_classes = np.argmax(predictions, axis=1)
    y_valid = np.argmax(y_valid, axis=-1)
    report = classification_report(y_valid, predicted_classes, target_names=le.classes_)
    classification_report_csv(report, 'Data/report/report.csv')
    print(report)
    confusion = confusion_matrix(y_valid, predicted_classes)

