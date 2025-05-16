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

from typing import Optional as Opt
import numpy.typing as npt
from tensorflow.data import Dataset as ds
from tensorflow.python.framework.ops import EagerTensor as eagert

from detectarrow.conf.paths import DATASET_PATH
from detectarrow.conf.paths import ARROWS_PATH
from detectarrow.conf.paths import MODELS_PATH
from detectarrow.conf.paths import MODEL_BNAME
from detectarrow.conf.imgs import COMPARED_SIZE
from detectarrow.processing.utils import get_newest_fname_in_path


class ModelHandler:
    def __init__(self):
        self.reset()

    # noinspection PyAttributeOutsideInit
    def reset(self) -> None:
        self.train_ds = None
        self.val_ds = None
        self.model = None
        self.history = None
        self.images = None
        self.labels = None

    def load_model(self, model_bname: Opt[str] = None) -> None:
        if model_bname is None:
            model_fname = get_newest_fname_in_path(MODELS_PATH)
        else:
            model_fname = str(PurePath(MODELS_PATH, model_bname))
        self.model = load_model(model_fname)

    def save_model(self, model_bname: str = None) -> None:
        if model_bname is None:
            model_fname = str(PurePath(MODELS_PATH, MODEL_BNAME))  # only overwrite default by default
        else:
            model_fname = str(PurePath(MODELS_PATH, model_bname))
        self.model.save(model_fname)

    def build_model(self) -> None:
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

    def load_dataset(self, dataset_path: str = DATASET_PATH, seed: int = 42, val_split: float = 0.2) -> None:
        self.train_ds = image_dataset_from_directory(
            dataset_path,
            label_mode='binary',
            color_mode='grayscale',
            batch_size=1_000,
            image_size=COMPARED_SIZE,
            seed=seed,
            validation_split=val_split,
            subset='training',
        )
        self.val_ds = image_dataset_from_directory(
            dataset_path,
            label_mode='binary',
            color_mode='grayscale',
            batch_size=1_000,
            image_size=COMPARED_SIZE,
            seed=seed,
            shuffle=True,
            validation_split=val_split,
            subset='validation',
        )

    def train_model(self, epochs: int = 10, train_ds: ds = None, val_ds: ds = None) -> None:
        if train_ds is not None:
            self.train_ds = train_ds
        if val_ds is not None:
            self.val_ds = val_ds
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs
        )

    def show_training_progress(self) -> None:
        loss_values = self.history.history.get('loss', None)
        val_loss_values = self.history.history.get('val_loss', None)
        plt_epochs = range(1, (len(loss_values) + 1))

        if val_loss_values:
            line1 = plt.plot(plt_epochs, val_loss_values, label='Validation/Test Loss')
            plt.setp(line1, linewidth=2.0, marker='+', markersize=10.0)
        if loss_values:
            line2 = plt.plot(plt_epochs, loss_values, label='Training Loss')
            plt.setp(line2, linewidth=2.0, marker='4', markersize=10.0)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.show()

        acc_values = self.history.history.get('accuracy', None)
        val_acc_values = self.history.history.get('val_accuracy', None)

        if val_acc_values:
            line1 = plt.plot(plt_epochs, val_acc_values, label='Validation/Test Accuracy')
            plt.setp(line1, linewidth=2.0, marker='+', markersize=10.0)
        if acc_values:
            line2 = plt.plot(plt_epochs, acc_values, label='Training Accuracy')
            plt.setp(line2, linewidth=2.0, marker='4', markersize=10.0)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        plt.show()

    def try_val_sample(self) -> None:
        test_image = None
        test_label = None
        for images, labels in self.val_ds.take(1):
            test_image = images[0].numpy()
            test_label = labels[0].numpy()
            break

        prediction = self.model.predict(test_image[None])
        print(f'prediction:{prediction[0][0]:.2f}')
        print(f'real:{test_label}')
        plt.imshow(test_image, cmap='gray')
        plt.show()

    def prepare_validation(self) -> None:
        res_imgs = []
        res_labels = []
        for images, labels in self.val_ds:
            res_imgs.extend(images)
            res_labels.extend(labels)

        self.images = np.array(res_imgs)
        self.labels = np.concatenate(res_labels, axis=0)

    def classification_report(self) -> None:
        predictions = (self.model.predict(self.images) > 0.5).astype("int32")
        print(classification_report(self.labels, predictions))

    def tensorflow_import_hack(self) -> None:
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

    def saliency_output_hack(self) -> None:
        layers = [layer.name for layer in self.model.layers]
        self.model.output_names = [layers[-1]]

    def saliency(self, import_hack: bool = False) -> None:
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
        labels = np.hstack((np.ones(5), np.zeros(5)))
        if import_hack:
            self.tensorflow_import_hack()

        from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
        from tf_keras_vis.saliency import Saliency

        replace2linear = ReplaceToLinear()

        def score_function(output: eagert) -> tuple[
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32
        ]:
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

        self.saliency_output_hack()  # https://github.com/onnx/tensorflow-onnx/issues/2319
        saliency = Saliency(
            self.model,
            model_modifier=replace2linear,
            clone=False
        )
        saliency_map = saliency(score_function, saliency_part)
        # image_titles = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        f, ax = plt.subplots(nrows=2, ncols=len(labels), figsize=(12, 4))
        for i, title in enumerate(labels):
            ax[0, i].set_title(title, fontsize=16)
            ax[0, i].imshow(saliency_map[i], cmap='jet')
            ax[0, i].axis('off')
            ax[1, i].imshow(saliency_part[i], cmap='gray')
            ax[1, i].axis('off')
        plt.tight_layout()
        plt.show()
