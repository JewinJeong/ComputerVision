import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Activation, ZeroPadding2D, Dropout, BatchNormalization

import tensorflow as tf
import os
import glob
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.debugging.set_log_device_placement(True)
# tf.compat.v2.logging.set_verbosity(tf.compat.v2.logging.Error)

def plot_loss_curve(history):

    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 10))

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()

def loadData():
    categories = ['food', 'interior', 'exterior']
    nb_class = len(categories)

    data_dir = "./images"
    batch_size = 10
    img_height = 300
    img_width = 300

    x = []
    y = []

    for idx, c in enumerate(categories):
        label = [0 for i in range(nb_class)]
        label[idx] = 1

        image_dir = data_dir + "/" + c
        files = glob.glob(image_dir + "/*.jpg")

        for i, f in enumerate(files):
            img = Image.open(f)
            img = img.convert("RGB")
            img = img.resize((img_width, img_height))
            data = np.asarray(img)
            x.append(data)
            y.append(label)
    x = np.array(x)
    y = np.array(y)

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)
    image_data = (X_train, X_test, Y_train, Y_test)
    np.save("./image_data.npy", image_data)

def train_model(X_train, X_test, Y_train, Y_test):
    # X_train, X_test, Y_train, Y_test = np.load("./image_data.npy", allow_pickle=True)
    print(Y_train)
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)

    model = Sequential([
        Input(shape=(300, 300, 3), name='input_layer'),
        Conv2D(96, kernel_size=11,strides=4, activation='relu', name='conv_layer1'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(256, kernel_size=5, activation='relu', name='conv_layer2'),
        ZeroPadding2D(),
        MaxPooling2D(pool_size=2),
        BatchNormalization(),
        Conv2D(384, kernel_size=3, activation='relu', name='conv_layer3'),
        ZeroPadding2D(),
        Conv2D(384, kernel_size=3, activation='relu', name='conv_layer4'),
        ZeroPadding2D(),
        Conv2D(256, kernel_size=3, activation='relu', name='conv_layer5'),
        ZeroPadding2D(),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(1024, activation='relu'),
        Dense(3, activation='softmax', name='output_layer')
    ])


    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=20, epochs=20)
    plot_loss_curve(history.history)
    print(history.history)
    print("train loss=", history.history['loss'][-1])
    print("validation loss=", history.history['val_loss'][-1])


    model.save('model-201611297')

    return model


def predict_image_sample(model, X_test, y_test, test_id=-1):
    tp = 0
    for i in range(9000):
        # if test_id < 0:
        #     from random import randrange
        #     test_sample_id = randrange(9000)
        # else:
        #     test_sample_id = test_id
        #
        test_image = X_test[i]
        # plt.imshow(test_image, cmap='gray')
        test_image = test_image.reshape(1, 300, 300, 3)
        y_actual = y_test[i]
        y_act = y_actual.argmax()
        # print("y_actual number=", y_actual)
        y_pred = model.predict(test_image)

        # print("y_pred=", y_pred)
        y_pre = np.argmax(y_pred, axis=1)[0]
        # print("y_pred number=", y_pred)
        if y_act == y_pre:
            tp += 1
    acc = tp / 9000
    print(acc)

if __name__ == '__main__':
    # loadData()
    X_train, X_test, Y_train, Y_test = np.load("./image_data.npy", allow_pickle=True)
    # model = load_model('model-201611297')
    model = train_model(X_train, X_test, Y_train, Y_test)
    predict_image_sample(model, X_test, Y_test)

    # model = Sequential()
    # img_shape = (300, 300, 3)

    # no_of_classes = 3
    #
    # # 레이어 1
    # model.add(Conv2D(96, (11, 11), input_shape=img_shape, padding='same', name='conv_layer1'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # # 레이어 2
    # model.add(Conv2D(256, (5, 5), padding='same', name='conv_layer2'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # # 레이어 3
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Conv2D(384, (3, 3), padding='same', name='conv_layer3'))
    # model.add(Activation('relu'))
    #
    # # 레이어 4
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Conv2D(384, (3, 3), padding='same', name='conv_layer4'))
    # model.add(Activation('relu'))
    #
    # # 레이어 5
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Conv2D(256, (3, 3), padding='same', name='conv_layer5'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # # 레이어 6
    # model.add(Flatten())
    # model.add(Dense(4096))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    #
    # # 레이어 7
    # model.add(Dense(4096))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    #
    # # 레이어 8
    # model.add(Dense(no_of_classes))
    # model.add(Activation('softmax'))