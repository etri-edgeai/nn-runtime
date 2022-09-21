# functions defined in etb package.

def build(docker_img_addr, src_path, arguments=None, work_path="."):
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
            arguments: list, a list of arguments (strings) will be given to the CLI program (your program supposed to be
                executed in an edge device)
            src_path: str, a root path to the CLI program
            work_path str, a working path will be given to Docker as the value of the '-f-' option.
            (You may omit this.)

    """

def run(nodes, timeout=None):
    """Run function (Actual execution)

        Args.

            nodes: list, a list of node ids (strings)
            timeout: int, the maximum waiting time (secs)

    """

def get_nodes():
    """Retrieve nodes names

        Returns.

            a list of node ids (strings)

    """

def wait_results():
    """Wait for the ETB process

        Returns.

            The output JSON string

    """

#######################################
def example():
    import etb
    nodes = etb.get_nodes()
    etb.build("129.254.165.171.5000/tensorrt_onnx:v1.0", "./src", arguments=["arg1", "arg2"])
    etb.run(nodes[:1], timeout=600)
    ret_json = etb.wait_results()
