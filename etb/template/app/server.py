from __future__ import print_function

import hashlib
import time
import os
import traceback

import numpy as np
import tensorflow as tf
from tensorflow import keras

from etb import etb

def gen_dockerfile(key):
    image_name = "192.168.0.2:5000/etb2:r32.6.1-tf2.6-py3"
    src_path = "./src"
    docker_file = f"""
FROM {image_name}
WORKDIR /usr/src/app
COPY {src_path} .
CMD [ "python3", "./test.py", "{key}"]
"""
    return docker_file

def run():

    test_jetson = True
    test_raspberry = False

    model = tf.keras.applications.efficientnet.EfficientNetB0(
        include_top=False, input_tensor=None, weights=None,
        input_shape=(32, 32, 3), pooling=None, classes=10,
        classifier_activation='softmax'
    )
    model.save("./src/model/m.h5")
    print("efficientnet test")

    try:
        if test_jetson:
            key = hashlib.md5(bytes(str(time.time()), 'utf-8')).hexdigest()

            # nvidia: linux/arm64
            etb.build("model2", "efficientnet", gen_dockerfile(key), platform="linux/arm64", work_path=".")
            etb.run("model2", "efficientnet", nodes=["n1"])

            reports = etb.wait_result("model2", "efficientnet", key=key, remove=True)
            #reports = etb.reports("model2", "efficientnet")
            print(reports)

    except Exception as ex:
        traceback.print_exc()
        print(ex)


if __name__ == "__main__":

    while True:
        run()
