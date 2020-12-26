import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, AveragePooling2D, Add,  Activation, ZeroPadding2D, Dropout, BatchNormalization
from tensorflow.keras.regularizers import L2

import glob
from PIL import Image
import os
import tensorflow as tf
import warnings

'''
    Image Classification 
    Food, interior, exterior
    
'''


warnings.filterwarnings(action='ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.debugging.set_log_device_placement(True)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.Error)

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

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, shuffle=True, stratify=y)
    image_data = (X_train, X_test, Y_train, Y_test)
    np.save("./image_data.npy", image_data)


def res_identity(x, filters):
  #renet block where dimension doesnot change.
  #The skip connection is just simple identity conncection
  #we will have 3 blocks and then input will be added

  x_skip = x # this will be used for addition with the residual block
  f1, f2 = filters

  #first block
  x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=L2(0.001))(x)
  x = BatchNormalization()(x)
  x = Activation(activation='relu')(x)

  #second block # bottleneck (but size kept same with padding)
  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(0.001))(x)
  x = BatchNormalization()(x)
  x = Activation(activation='relu')(x)

  # third block activation used after adding the input
  x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=L2(0.001))(x)
  x = BatchNormalization()(x)
  # x = Activation(activations.relu)(x)

  # add the input
  x = Add()([x, x_skip])
  x = Activation(activation='relu')(x)

  return x

def res_conv(x, s, filters):
  '''
  here the input size changes'''
  x_skip = x
  f1, f2 = filters

  # first block
  x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=L2(0.001))(x)
  # when s = 2 then it is like downsizing the feature map
  x = BatchNormalization()(x)
  x = Activation(activation='relu')(x)

  # second block
  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(0.001))(x)
  x = BatchNormalization()(x)
  x = Activation(activation='relu')(x)

  #third block
  x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=L2(0.001))(x)
  x = BatchNormalization()(x)

  # shortcut
  x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=L2(0.001))(x_skip)
  x_skip = BatchNormalization()(x_skip)

  # add
  x = Add()([x, x_skip])
  x = Activation(activation='relu')(x)

  return x

def resnet50(X_train, X_test, Y_train, Y_test):

  input_im = Input(shape=(300,300,3)) # cifar 10 images size
  x = ZeroPadding2D(padding=(3, 3))(input_im)

  # 1st stage
  # here we perform maxpooling, see the figure above

  x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
  x = BatchNormalization()(x)
  x = Activation(activation='relu')(x)
  x = MaxPooling2D((3, 3), strides=(2, 2))(x)

  #2nd stage
  # frm here on only conv block and identity block, no pooling

  x = res_conv(x, s=1, filters=(64, 256))
  x = res_identity(x, filters=(64, 256))
  x = res_identity(x, filters=(64, 256))

  # 3rd stage

  x = res_conv(x, s=2, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))

  # 4th stage

  x = res_conv(x, s=2, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))

  # 5th stage

  x = res_conv(x, s=2, filters=(512, 2048))
  x = res_identity(x, filters=(512, 2048))
  x = res_identity(x, filters=(512, 2048))

  # ends with average pooling and dense connection

  x = AveragePooling2D((2, 2), padding='same')(x)

  x = Flatten()(x)
  x = Dense(3, activation='softmax', kernel_initializer='he_normal')(x) #multi-class

  # define the model

  model = Model(inputs=input_im, outputs=x, name='Resnet50')

  model.summary()

  model.compile(optimizer='Nadam', loss='categorical_crossentropy', metrics=['accuracy'])

  history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=30)
  plot_loss_curve(history.history)
  print(history.history)
  print("train loss=", history.history['loss'][-1])
  print("validation loss=", history.history['val_loss'][-1])

  model.save('./model-201611297')

  return model



def train_model(X_train, X_test, Y_train, Y_test):
    # X_train, X_test, Y_train, Y_test = np.load("./image_data.npy", allow_pickle=True)
    model = Sequential([
        Input(shape=(300, 300, 3), name='input_layer'),
        Conv2D(16, kernel_size=(3, 3), activation='relu', name='conv_layer1'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, kernel_size=(3, 3), activation='relu', name='conv_layer2'),
        Dropout(0.2),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv_layer3'),
        MaxPooling2D(pool_size=(2, 2)),
        # Conv2D(20, kernel_size=2, activation='relu', name='conv_layer4'),
        # MaxPooling2D(pool_size=2),
        Flatten(),
        #  Dense(64, activation='relu', name="dense_layer")
        Dense(3, activation='softmax', name='output_layer')
    ])
    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=20, epochs=30)
    plot_loss_curve(history.history)
    print(history.history)
    print("train loss=", history.history['loss'][-1])
    print("validation loss=", history.history['val_loss'][-1])

    model.save('model-201611297-2')


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
    # model = load_model('model-201611297-12143')
    model = train_model(X_train, X_test, Y_train, Y_test)
    predict_image_sample(model, X_test, Y_test)



