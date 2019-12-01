from keras.models import load_model
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from mylib.classification_report import classification_report_csv
from mylib.datasets import SimpleDatasetLoader
from mylib.preprocessing import SimplePreprocessor
from mylib.preprocessing import PaddingPreprocessor
from sklearn.model_selection import StratifiedKFold
from imutils import paths
from sklearn import preprocessing
import numpy as np


k = 10

print("[INFO] Loading Images...")
imagePaths = paths.list_images("Data/images")
sp = SimplePreprocessor(32, 32)
pp = PaddingPreprocessor(48, 48, 48, 48, borderValue=0)
sdl = SimpleDatasetLoader(preprocessors=[sp, pp])
(X, Y) = sdl.load(imagePaths)

# Label Encoding
le = preprocessing.LabelEncoder()
Y = le.fit_transform(Y)

skf = StratifiedKFold(n_splits=k, random_state=50)
folds = skf.split(X, Y)
Y = to_categorical(Y)
for i, (train_index, test_index) in enumerate(folds):
    print('[INFO] Fold=', i)
    train_X, test_X = X[train_index], X[test_index]
    train_Y, test_Y = Y[train_index], Y[test_index]
    model_path = "Data/Model/stratified k fold/model_fold_{}.h5".format(i)
    # model_path = "Data/Model/fold_{}_model.hdf5".format(i)
    model = load_model(model_path)
    predictions = model.predict(test_X, batch_size=32)
    predicted_classes = np.argmax(predictions, axis=1)
    test_Y = np.argmax(test_Y, axis=-1)
    print("[INFO] Classification report")
    report = classification_report(test_Y, predicted_classes, target_names=le.classes_)
    classification_report_csv(report, 'Data/report/classification_report/report_fold_{}.csv'.format(i))
    print("[INFO] Confusion matrix")
    confusion = confusion_matrix(test_Y, predicted_classes)
    cm = np.asarray(confusion)
    cm_path = "Data/report/confusion_matrix/confusion_matrix_fold_{}.csv".format(i)
    np.savetxt(cm_path, cm, delimiter=",")
