import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir: str) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    data_dir_path = Path(data_dir)
    images: List[np.ndarray] = []
    labels: List[int] = []

    for sign_dir_path in data_dir_path.iterdir():
        print(f"Reading from {sign_dir_path}")
        sign_id = int(sign_dir_path.name)
        for sign_img_path in sign_dir_path.iterdir():
            # Append label
            labels.append(sign_id)
            # Load image
            img: np.ndarray = cv2.imread(str(sign_img_path))
            # Check dimensions
            width, height, channel = img.shape
            # Each pixel must have 3 channels
            assert channel == 3
            # Height too big, crop sides
            if height > IMG_HEIGHT:
                start = (height - IMG_HEIGHT) // 2
                end = start+IMG_HEIGHT
                img = img[:, start:end, :]
            # Height too small, pad
            elif height < IMG_HEIGHT:
                before = (IMG_HEIGHT - height) // 2
                after = IMG_HEIGHT - height - before
                img = np.pad(img, [[0, 0], [before, after], [0, 0]])
            # Width too big, crop sides
            if width > IMG_WIDTH:
                start = (width - IMG_WIDTH) // 2
                end = start + IMG_WIDTH
                img = img[start:end, :, :]
            # Width too small, pad
            elif width < IMG_WIDTH:
                before = (IMG_WIDTH - width) // 2
                after = IMG_WIDTH - width - before
                img = np.pad(img, [[before, after], [0, 0], [0, 0]])
            # Check shape
            assert img.shape == (IMG_WIDTH, IMG_HEIGHT, 3)
            # Append picture
            images.append(img)

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Common activation functions
    relu = tf.keras.activations.relu
    softmax = tf.keras.activations.softmax
    # Common regularizer
    l2 = tf.keras.regularizers.l2(l2=1e-5)
    # Sequential model initialization
    model = tf.keras.models.Sequential()
    # Input 1×30×30×3
    model.add(tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    # 1×30×30×3 => 36×26×26×3 (5×5 Conv2D)
    model.add(tf.keras.layers.Conv2D(36, (5, 5), activation=relu))
    # 36×26×26×3 => 36×13×13×3 (2×2 MaxPool2D)
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    # 36×13×13×3 => 96×10×10×3 (4×4 Conv2D)
    model.add(tf.keras.layers.Conv2D(96, (4, 4), activation=relu))
    # 96×10×10×3 => 96×5×5×3 (2×2 MaxPool2D)
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    # 96×5×5×3 => 7200 (Flatten)
    model.add(tf.keras.layers.Flatten())
    # 7200 => 2500 (Linear)
    model.add(tf.keras.layers.Dense(
        2500, activation=relu, kernel_regularizer=l2))
    model.add(tf.keras.layers.Dropout(0.2))
    # 2500 => 1200 (Linear)
    model.add(tf.keras.layers.Dense(
        1200, activation=relu, kernel_regularizer=l2))
    model.add(tf.keras.layers.Dropout(0.2))
    # 1200 => 400 (Linear)
    model.add(tf.keras.layers.Dense(
        400, activation=relu, kernel_regularizer=l2))
    model.add(tf.keras.layers.Dropout(0.1))
    # 400 => 160 (Linear)
    model.add(tf.keras.layers.Dense(
        160, activation=relu, kernel_regularizer=l2))
    model.add(tf.keras.layers.Dropout(0.1))
    # 160 => 43
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation=softmax))
    # Compilation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=[tf.keras.metrics.Accuracy()]
    )
    return model


if __name__ == "__main__":
    main()
