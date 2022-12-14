import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D, Flatten, BatchNormalization
from tensorflow.keras.applications.efficientnet import EfficientNetB0

def get_model(input_shape, n_classes=100):
    efnb0 = EfficientNetB0(
        include_top=False, weights='imagenet', input_shape=input_shape)

    model = Sequential()
    model.add(efnb0)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    return model

def test():
    model = get_model((224, 224, 3))
    model.summary()

if __name__ == "__main__":
    test()
