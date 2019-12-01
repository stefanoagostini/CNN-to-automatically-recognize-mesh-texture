from mylib.datasets import SimpleDatasetLoader
from mylib.preprocessing import SimplePreprocessor
from mylib.preprocessing import PaddingPreprocessor
from keras import backend as K
from keras.applications import MobileNetV2
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.layers.pooling import AveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
from imutils import paths
from mylib.callbacks import TrainingMonitor
from keras.callbacks import ModelCheckpoint

import os
import argparse


def get_model():
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(128, 128, 3)))
    head_model = base_model.output
    head_model = AveragePooling2D(pool_size=(4, 4))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(128, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(13, activation="softmax")(head_model)  # 13 labels if there are 10 models
    model = Model(inputs=base_model.input, outputs=head_model)
    opt = Adam(lr=1e-4)  # tried from -3 to -6
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


if __name__ == '__main__':
    # argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--split-policy", choices=['single', 'kfold'], default='kfold')
    ap.add_argument("-v", "--validation-split", default='0.2', help="validation split")
    ap.add_argument("-k", "--k-fold", default='10', help="number of folds for k-fold cross validation")
    ap.add_argument("-b", "--batch-size", default='32', help="batch size")
    ap.add_argument("-e", "--epochs", default='20', help="number of training epochs")
    args = vars(ap.parse_args())
    split_policy = args['split_policy']
    batch_size = int(args['batch_size'])
    val_split_percent = float(args['validation_split'])
    epochs = int(args['epochs'])
    k = int(args["k_fold"])

    print("[INFO] Loading Images...")
    imagePaths = paths.list_images("Data/images")
    sp = SimplePreprocessor(32, 32)
    pp = PaddingPreprocessor(48, 48, 48, 48, borderValue=0)
    sdl = SimpleDatasetLoader(preprocessors=[sp, pp])
    (X, Y) = sdl.load(imagePaths)

    # Label Encoding
    le = preprocessing.LabelEncoder()
    Y = le.fit_transform(Y)

    if split_policy == 'single':
        Y = to_categorical(Y)
        x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=val_split_percent, random_state=50)

        print("[INFO] Training...")
        model = get_model()
        print(model.summary())

        # model checkpoint
        model_path = "Data/Callbacks/single_test_model.h5"
        checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=True, mode='min')

        # build the path to the training plot and training history
        plotPath = os.path.sep.join(["Data/Callbacks", "training_plot.png"])
        jsonPath = os.path.sep.join(["Data/Callbacks", "training_history.json"])
        # history checkpoint
        training_monitor = TrainingMonitor(plotPath, jsonPath=jsonPath)
        callbacks_list = [checkpoint, training_monitor]

        # fit
        history = model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_valid, y_valid), epochs=epochs, verbose=2, callbacks=callbacks_list)
        model.save('Data/Model/model_single.hdf5')

    if split_policy == 'kfold':
        skf = StratifiedKFold(n_splits=k, random_state=50)
        folds = skf.split(X, Y)
        Y = to_categorical(Y)
        for i, (train_index, test_index) in enumerate(folds):
            print('[INFO] Fold=', i)
            train_X, test_X = X[train_index], X[test_index]
            train_Y, test_Y = Y[train_index], Y[test_index]
            K.clear_session()
            model = get_model()
            # model checkpoint
            model_path = "Data/Model/fold_{}_model.hdf5".format(i)
            checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
            callbacks_list = [checkpoint]
            history = model.fit(train_X, train_Y, batch_size=batch_size, validation_split=0.2,
                                epochs=epochs, verbose=2)
            model.load_weights(model_path)
            print('Model evaluation: ', model.evaluate(test_X, test_Y))



