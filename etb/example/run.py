from __future__ import print_function

import hashlib
import time
import os
import traceback

import numpy as np
import tensorflow as tf
from tensorflow import keras

from etb import etb_apis

def _build(key):
    #image_name = "nvcr.io/nvidia/l4t-tensorflow:r32.7.1-tf2.7-py3"
    #image_name = "nvcr.io/nvidia/tensorrt:23.02-py3"
    image_name = "nvcr.io/nvidia/tensorflow:23.03-tf2-py3"
    src_path = "./src"
    cmd = f"""[ "python3", "./test.py", "{key}" ]"""
    etb_apis.build(image_name, src_path, cmd, platform="linux/x86_64")

def run():

    model = tf.keras.applications.efficientnet.EfficientNetB0(
        include_top=False, input_tensor=None, weights=None,
        input_shape=(32, 32, 3), pooling=None, classes=10,
        classifier_activation='softmax'
    )
    model.save("./src/model/m.h5")
    print("efficientnet test")

    ####### Implement here! ########
    # get/print the inference time
    # test_app runs without evaluation
    ################################
    try:
        key = hashlib.md5(bytes(str(time.time()), 'utf-8')).hexdigest()

        # nvidia: linux/arm64
        _build(key)
        tid = etb_apis.run(nodes=['n1'])
        reports = etb_apis.wait_result(key)
        etb_apis.download(tid, filename="tempa.zip")
        print(reports)

    except Exception as ex:
        traceback.print_exc()
        print(ex)


if __name__ == "__main__":

    run()
