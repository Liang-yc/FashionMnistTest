
from __future__ import print_function
from __future__ import division

import os

import numpy as np
import sklearn.metrics as metrics
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,EarlyStopping
from keras.datasets import cifar10
from keras.optimizers import Adam,SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

from resnext import ResNext
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from keras.layers.advanced_activations import PReLU
from inception_v4 import create_model
from shake_shake_model import create_shakeshake_cifar
from random_eraser import get_random_eraser  # added


def train_model_with_2_classes(model,train_X,train_Y,test_X,test_Y):
    train_X = (train_X-0.1307)/0.3081
    test_X = (test_X-0.1307)/0.3081
    id_list=[]

    out_dir = "models/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i in range(2,3):
        for j in range(8,10):
            id_list.append([i,j])
            two_classes_id=(train_Y==i)+ (train_Y==j)
            trainX=train_X[two_classes_id]
            trainY=train_Y[two_classes_id]

            two_classes_id=(test_Y==i)+( test_Y==j)
            testX=test_X[two_classes_id]
            testY=test_Y[two_classes_id]

            Y_train = np_utils.to_categorical(trainY, nb_classes)
            Y_test = np_utils.to_categorical(testY, nb_classes)


            # Load model
            weights_file = "models/shake_shake"+str(i)+'_'+str(j)+".h5"
            if os.path.exists(weights_file):
                model.load_weights(weights_file)
                print(weights_file,"Model loaded.\r\n")

            lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5,  # 当标准评估停止提升时，降低学习速率。
                                           cooldown=0, patience=20, min_lr=1e-8)

            model_checkpoint = ModelCheckpoint(weights_file, monitor="val_acc", save_best_only=True,
                                               save_weights_only=True, mode='auto')
            earlyStopping = EarlyStopping(monitor='acc', patience=10, verbose=1, mode='auto')
            callbacks = [lr_reducer, model_checkpoint,earlyStopping]

            train_data = ImageDataGenerator(featurewise_center=True,
                                            featurewise_std_normalization=True,
                                            # preprocessing_function=random_crop_image,
                                            preprocessing_function=get_random_eraser(v_l=0, v_h=1),
                                            rotation_range=10,
                                            width_shift_range=5. / 28,
                                            height_shift_range=5. / 28,
                                            horizontal_flip=True)
            validation_data = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
            for data in (train_data, validation_data):
                data.fit(trainX)
            model.fit_generator(train_data.flow(trainX, Y_train, batch_size=batch_size),
                            steps_per_epoch=len(trainX) // batch_size,
                            epochs=nb_epoch,
                            callbacks=callbacks,
                            validation_data=validation_data.flow(testX, Y_test, batch_size=batch_size),
                            validation_steps=testX.shape[0] // batch_size, verbose=1)
            # model.fit(trainX, Y_train, batch_size=batch_size, epochs=1, verbose=1, validation_data=(testX, Y_test),
            #           callbacks=callbacks)
            yPreds = model.predict(testX)
            yPred = np.argmax(yPreds, axis=1)
            yTrue = testY

            accuracy = metrics.accuracy_score(yTrue, yPred) * 100
            error = 100 - accuracy
            print(i,'pk',j)
            print("Accuracy : ", accuracy)
            print("Error : ", error)

if __name__ == '__main__':


    batch_size = 100
    nb_classes = 10
    nb_epoch = 100

    img_rows, img_cols = 28,28
    img_channels = 1




    # Import data
    mnist = read_data_sets('./data/fashion', reshape=False, validation_size=0,
                           source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
    trainX = mnist.train.images
    trainY = mnist.train.labels
    testX= mnist.test.images
    testY = mnist.test.labels
    id = trainY==0
    part_id = trainY[id]
    Y_train = np_utils.to_categorical(trainY, nb_classes)
    Y_test = np_utils.to_categorical(testY, nb_classes)

    img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)
    depth = 20
    cardinality = 8
    width = 16

    # model = ResNext(img_dim, depth=depth, cardinality=cardinality, width=width, weights=None, classes=nb_classes)
    model=create_shakeshake_cifar(n_classes=10)
    print("Model created")

    model.summary()

    optimizer = Adam(lr=3e-4)  # Using Adam instead of SGD to speed up training
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    print("Finished compiling")
    print("Building model...")
    train_model_with_2_classes(model,trainX, trainY, testX, testY)


    # train_data = ImageDataGenerator(featurewise_center=True,
    #                                 featurewise_std_normalization=True,
    #                                 # preprocessing_function=random_crop_image,
    #                                 preprocessing_function=get_random_eraser(v_l=0, v_h=1),
    #                                 rotation_range=10,
    #                                 width_shift_range=5. / 28,
    #                                 height_shift_range=5. / 28,
    #                                 horizontal_flip=True)
    # validation_data = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,preprocessing_function=get_random_eraser(istest=True))
    # for data in (train_data, validation_data):
    #     data.fit(trainX)
    # generator.fit(trainX, seed=0)


    # os.system('shutdown -s -f -t 59')