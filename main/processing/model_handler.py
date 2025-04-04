# noinspection PyUnresolvedReferences
from keras.preprocessing import image_dataset_from_directory
# noinspection PyUnresolvedReferences
from keras.models import load_model
# noinspection PyUnresolvedReferences
from keras.models import Sequential
# noinspection PyUnresolvedReferences
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Rescaling, Input
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pathlib as pl
from pathlib import PurePath
from sklearn.metrics import classification_report

import numpy.typing as npt

from main.conf.paths import DATASET_PATH
from main.conf.paths import ARROWS_PATH
from main.conf.paths import MODEL_PATH
from main.conf.imgs import COMPARED_SIZE


class ModelHandler:
    def __init__(self):
        self.reset()

    # noinspection PyAttributeOutsideInit
    def reset(self):
        self.train_ds = None
        self.val_ds = None
        self.model = None
        self.history = None
        self.images = None
        self.labels = None

    def load_model(self, basename: str = 'arrow_detection.keras'):
        filepath = str(PurePath(MODEL_PATH, basename))
        try:
            self.model = load_model(filepath)
            return True
        except OSError:
            return False

    def save_model(self, basename: str = 'arrow_detection.keras'):
        if self.model is not None:
            filepath = str(PurePath(MODEL_PATH, basename))
            self.model.save(filepath)
            return True

        else:
            return False

    def build_model(self):
        self.model = Sequential([
            Input(shape=(68, 24, 1)),
            Rescaling(1. / 255),
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='same'),
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='same'),
            Flatten(),
            Dense(100, activation='relu'),
            Dropout(0.05),
            Dense(1, activation='sigmoid'),
        ])
        self.model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        print(self.model.summary())

    def load_datasets(self):
        self.train_ds = image_dataset_from_directory(
            DATASET_PATH,
            label_mode='binary',
            color_mode='grayscale',
            batch_size=1_000,
            image_size=COMPARED_SIZE,
            seed=42,
            validation_split=0.2,
            subset='training',
        )
        self.val_ds = image_dataset_from_directory(
            DATASET_PATH,
            label_mode='binary',
            color_mode='grayscale',
            batch_size=1_000,
            image_size=COMPARED_SIZE,
            seed=42,
            shuffle=True,
            validation_split=0.2,
            subset='validation',
        )

    def train_model(self, epochs=10):
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs
        )

    def show_training_progress(self):
        if self.history is not None:
            loss_values = self.history.history['loss']
            val_loss_values = self.history.history['val_loss']
            plt_epochs = range(1, (len(loss_values) + 1))

            line1 = plt.plot(plt_epochs, val_loss_values, label='Validation/Test Loss')
            line2 = plt.plot(plt_epochs, loss_values, label='Training Loss')
            plt.setp(line1, linewidth=2.0, marker='+', markersize=10.0)
            plt.setp(line2, linewidth=2.0, marker='4', markersize=10.0)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            plt.show()

            acc_values = self.history.history['accuracy']
            val_acc_values = self.history.history['val_accuracy']

            line1 = plt.plot(plt_epochs, val_acc_values, label='Validation/Test Accuracy')
            line2 = plt.plot(plt_epochs, acc_values, label='Training Accuracy')
            plt.setp(line1, linewidth=2.0, marker='+', markersize=10.0)
            plt.setp(line2, linewidth=2.0, marker='4', markersize=10.0)
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.grid(True)
            plt.legend()
            plt.show()
            return True

        else:
            return False

    def try_val_sample(self):
        if self.val_ds is not None:
            test_image = None
            for images, labels in self.val_ds.take(1):
                test_image = images[0].numpy()
                test_label = labels[0].numpy()
                break

            if test_image is not None:
                prediction = self.model.predict(test_image[None])
                print(f'prediction:{prediction[0][0]:.2f}')
                print(f'real:{test_label}')
                plt.imshow(test_image, cmap='gray')
                plt.show()
                return True

            else:
                return False

        else:
            return False

    def prepare_validation(self):
        res_imgs = []
        res_labels = []

        for images, labels in self.val_ds:
            res_imgs.extend(images)
            res_labels.extend(labels)

        self.images = np.array(res_imgs)
        self.labels = np.concatenate(res_labels, axis=0)

    def classification_report(self):
        if self.model is not None:
            if self.images is None or self.labels is None:
                self.prepare_validation()
            predictions = (self.model.predict(self.images) > 0.5).astype("int32")
            print(classification_report(self.labels, predictions))
            return True

        else:
            return False

    def do_quick_hack(self):
        try:
            import tensorflow.keras
        except ImportError:
            from tensorflow import keras
            import tensorflow

            import sys

            tensorflow.keras = keras
            tensorflow.keras.backend = keras.backend

            sys.modules['tensorflow.keras'] = sys.modules['keras']
            sys.modules['tensorflow.keras.backend'] = 'keras hack'

    def saliency(self, quick_hack=False):
        if self.model is None:
            return False

        if self.images is None or self.labels is None:
            self.prepare_validation()

        saliency_part = [0] * 10
        counter = 0
        for num in range(len(self.labels)):
            if self.labels[num] == 1:
                saliency_part[counter] = self.images[num]
                counter += 1
                if counter == 5:
                    break

        for num in range(len(self.labels)):
            if self.labels[num] == 0:
                saliency_part[counter] = self.images[num]
                counter += 1
                if counter == 10:
                    break

        saliency_part = np.asarray(saliency_part)
        for i in saliency_part:
            plt.imshow(i, cmap='gray')
            plt.show()

        if quick_hack:
            self.do_quick_hack()

        from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
        from tf_keras_vis.saliency import Saliency

        replace2linear = ReplaceToLinear()

        def score_function(output):
            return (
                output[0][0],
                output[1][0],
                output[2][0],
                output[3][0],
                output[4][0],
                output[5][0],
                output[6][0],
                output[7][0],
                output[8][0],
                output[9][0]
            )

        saliency = Saliency(self.model,
                            model_modifier=replace2linear,
                            clone=False)

        saliency_map = saliency(score_function, saliency_part)
        image_titles = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        f, ax = plt.subplots(nrows=2, ncols=len(image_titles), figsize=(12, 4))
        for i, title in enumerate(image_titles):
            ax[0, i].set_title(title, fontsize=16)
            ax[0, i].imshow(saliency_map[i], cmap='jet')
            ax[0, i].axis('off')
            ax[1, i].imshow(saliency_part[i], cmap='gray')
            ax[1, i].axis('off')
        plt.tight_layout()
        plt.show()
