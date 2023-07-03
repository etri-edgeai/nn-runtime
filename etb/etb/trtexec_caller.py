# -*- coding: utf-8 -*-
# Call trtexec

from __future__ import print_function

import time
import sys
import shutil
import os
import json
import subprocess

def run():

    # This program has a key as the first argument.
    key = sys.argv[1]
    cmd = " ".join(sys.argv[2:])

    print(cmd)
    t1 = time.time()
    st, res = subprocess.getstatusoutput(cmd)
    if st != 0:
        pass  # do something!
    print(res)
    t2 = time.time()
    time_ = float(t2 - t1) / 10

    basename = "output_model.trt"

    # You can pass anything with `data` to the caller.
    data = {
        "meta": {
            "elapsed_time": time_
        },
        "key": key,
        "filename": basename # basename 
    }

    # 출력
    print("result:", json.dumps(data))
    # docker's /files <-> the local dir, named files of the agent.

    if not os.path.exists("/files"):
        os.mkdir("/files")
    shutil.copy(path, "/files/"+basename) # the model file (path) must be moved to /files

if __name__ == "__main__":
    run()
