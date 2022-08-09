import os, sys, pathlib, subprocess, shutil
from utils.general import file_size
_default_openvino_output_path = './temp/openvino_output'
_default_openvino_xml_file_name = 'output.xml'
_default_openvino_bin_file_name = 'output.bin'
_default_openvino_map_file_name = 'output.mapping'
_default_tensorflow_file_path = './temp/tensorflow_output/output.h5'

def export_onnx2openvino(onnx_path, output_dir=_default_openvino_output_path):
    """
       Converting onnx to openvino
       Returns openvino output path and an exception if an exception is raised.
    """
    try:
        #check_requirements(('openvino-dev',))  # requires openvino-dev: https://pypi.org/project/openvino-dev/
        import openvino.inference_engine as ie

        print(f'{__file__} starting export with openvino {ie.__version__}...')

        #os.makedirs('../opt/output', exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        cmd = f"mo --input_model {onnx_path} --output_dir {output_dir} --data_type FP32"
        subprocess.check_output(cmd.split())  # export

        print(f'{__file__} export success, saved as {output_dir} ({file_size(output_dir):.1f} MB)')
        file_list = os.listdir(output_dir)
        for ff in file_list:
            if pathlib.Path(ff).suffix == ".xml":
                tmp_path = os.path.join(output_dir, ff)
                os.rename(tmp_path, os.path.join(output_dir, _default_openvino_xml_file_name))
            elif pathlib.Path(ff).suffix == ".bin":
                tmp_path = os.path.join(output_dir, ff)
                os.rename(tmp_path, os.path.join(output_dir, _default_openvino_bin_file_name))
            else: # mapping
                tmp_path = os.path.join(output_dir, ff)
                os.rename(tmp_path, os.path.join(output_dir, _default_openvino_map_file_name))
        return output_dir, None
    except Exception as e:
        print(f'export failure: {e}')
        return output_dir, e
        
