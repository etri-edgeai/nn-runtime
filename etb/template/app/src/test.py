# -*- coding: utf-8 -*-

from __future__ import print_function

import time
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
import json


# load models
def load_model(path):
    model = tf.keras.models.load_model(path)
    return model


def run():

    key = sys.argv[1]

    num_classes = 100
    path = "model/m.h5"
    model = load_model(path)
    shape_ = list(model.input.shape)
    shape_[0] = 10
    rand_ = np.random.rand(*shape_)

    t1 = time.time()
    model.predict(rand_)
    t2 = time.time()
    time_ = float(t2 - t1) / 10

    params = model.count_params()
    data = {
        "label": {
            "epoch": 2000,
            "size": 2000,
            "iteration": 2000,
            "params": params,
            "tf_version": tf.__version__,
            "gpu": tf.test.is_gpu_available()
        },
        "metric": {
            "elapsed_time": time_
        },
        "key": key
    }

    # 출력
    print("result:", json.dumps(data))

if __name__ == "__main__":
    run()
