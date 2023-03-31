# functions defined in etb package.
from etb import etb

def build(docker_img_addr, src_path, arguments, platform="linux/arm64", work_path="."):
    """ Build function (Preparation)

        ** NOTE: Assume that we have a CLI program converting a (TF/Torch/ONNX) model to a TRT model.
                 (Of course, it also measures the inference time of the converted TRT model.)
                 The CLI program is supposed to have a string value as the first argument.
                 The output JSON string of the program should include the first argument with a key value named "key".
                 
                 Here is an example ouptut JSON string.
                 '''
                 { 
                    "data": {
                        "key": the_first_arg,
                        "metric":{
                            "elapsed_time": inference_time
                        }
                    }
                 }
                 '''

        Args.
        
            docker_img_addr: str, docker image address
            src_path: str, a root path to the CLI program
            arguments: list, a list of arguments (strings) will be given to the CLI program (your program supposed to be
                executed in an edge device)
            platform: str, default: "linux/arm64", which will be passed to the input for the `platform` option of `docker buildx`.
            work_path str, a working path will be given to Docker as the value of the '-f-' option.
            (You may omit this.)

    """
    cmd = str(arguments)
    docker_file = f"""
FROM {docker_img_addr}
WORKDIR /usr/src/app
COPY {src_path} .
CMD {cmd}
"""

    print(docker_file)
    etb.build("test", "v1.0", docker_file, platform=platform, work_path=work_path)

def run(nodes, timeout=None):
    """Run function (Actual execution)

        Args.

            nodes: list, a list of node ids (strings)
            timeout: int, the maximum waiting time (secs)

    """
    return etb.run("test", "v1.0", nodes=nodes, timeout=timeout)

def get_nodes():
    """Retrieve nodes names

        Returns.

            a list of node ids (strings)

    """
    return etb.nodes()

def wait_result(key):
    """Wait for the ETB process

        Returns.

            The output JSON string

    """
    return etb.wait_result("test", "v1.0", key, remove=True)

def download(tid, filename="temp.zip"):
    etb.download_file(tid, filename=filename)
