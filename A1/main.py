import click
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
import os


def train():
    batch_size = 128
    num_classes = 10
    epochs = 12

    img_rows, img_cols = 28, 28

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save("data/out/model")


def predict():
    check_model = os.path.exists("data/out/model")
    check_input = os.path.exists("data/input")
    if not check_input:
        raise Exception("Please provide data through input 'data/input' folder")
    if not check_model:
        raise Exception("Please run container with --train first")

    imgen = ImageDataGenerator()
    imgen_folder = imgen.flow_from_directory("data", target_size=(28, 28),
                                             color_mode="grayscale", shuffle=False,
                                             classes=["input"], batch_size=32)
    model = load_model("data/out/model")
    predictions = model.predict_generator(imgen_folder, steps=np.ceil(len(imgen_folder.filenames) / 32.0))
    predictions = [str(x).strip("()") + "\ngit" for x in zip(imgen_folder.filenames, np.argmax(predictions, axis=1))]
    print(predictions)
    with open("data/out/results.txt", "w+") as file:
        file.writelines(predictions)


@click.command()
@click.option('--mode', default="train", help='number of greetings')
def router(mode):
    click.echo('Container is running in %s mode.' % mode)
    if mode == "train":
        train()
    elif mode == "predict":
        predict()
    else:
        raise ValueError("Please provide either --train or --predict mode!")


if __name__ == '__main__':
    router()
