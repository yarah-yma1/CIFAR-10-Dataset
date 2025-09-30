import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model


# data class
class CIFAR10Data:
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()
        print(self.x_train.shape, self.y_train.shape, self.x_test.shape, self.y_test.shape)

        # normalize
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0
        # flatten labels
        self.y_train, self.y_test = self.y_train.flatten(), self.y_test.flatten()

    def visualize(self, num=25):
        fig, ax = plt.subplots(5, 5)
        k = 0
        for i in range(5):
            for j in range(5):
                ax[i][j].imshow(self.x_train[k], aspect='auto')
                k += 1
        plt.show()


# model bulding class
class CIFAR10Model:
    def __init__(self, input_shape, num_classes):
        self.model = self.build_model(input_shape, num_classes)

    def build_model(self, input_shape, num_classes):
        i = Input(shape=input_shape)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        x = Flatten()(x)
        x = Dropout(0.2)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(num_classes, activation='softmax')(x)

        model = Model(i, x)
        model.summary()
        return model

    def compile(self):
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, x_train, y_train, x_test, y_test, epochs=50):
        return self.model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=epochs
        )

    def train_with_augmentation(self, x_train, y_train, x_test, y_test, epochs=50, batch_size=32):
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True
        )
        train_gen = datagen.flow(x_train, y_train, batch_size)
        steps_per_epoch = x_train.shape[0] // batch_size

        return self.model.fit(
            train_gen,
            validation_data=(x_test, y_test),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs
        )

    def save(self, filename="cifar10_model.h5"):
        self.model.save(filename)

# eval + predict class
class ModelEvaluator:
    def __init__(self, model, labels):
        self.model = model
        self.labels = labels

    def plot_accuracy(self, history):
        plt.plot(history.history['accuracy'], label='acc', color='red')
        plt.plot(history.history['val_accuracy'], label='val_acc', color='green')
        plt.legend()
        plt.show()

    def predict_image(self, x_test, y_test, image_number=0):
        plt.imshow(x_test[image_number])
        plt.show()

        p = x_test[image_number].reshape(1, 32, 32, 3)
        predicted_label = self.labels[self.model.predict(p).argmax()]
        original_label = self.labels[y_test[image_number]]

        print(f"Original: {original_label}, Predicted: {predicted_label}")


# main class
if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)

    # load data
    data = CIFAR10Data()
    data.visualize()

    # number of classes
    K = len(set(data.y_train))
    print("Number of classes:", K)

    # build model
    model_wrapper = CIFAR10Model(data.x_train[0].shape, K)
    model_wrapper.compile()

    # train model
    history = model_wrapper.train(data.x_train, data.y_train, data.x_test, data.y_test, epochs=5)

    # evaluate
    labels = '''airplane automobile bird cat deer dog frog horse ship truck'''.split()
    evaluator = ModelEvaluator(model_wrapper.model, labels)
    evaluator.plot_accuracy(history)
    evaluator.predict_image(data.x_test, data.y_test)

    # save model
    model_wrapper.save()