import os
import random
import string
import shutil
import uuid
import logging
from enum import Enum
from fastapi import Form
from jinja2 import Environment, FileSystemLoader
from setuptools import setup
from typing import Optional
from pydantic import BaseModel, validator
from obfuscator.obfuscator import obfuscate

def make_template(env: Environment, target, dist, render_data:dict):
    template = env.get_template(target)
    with open(os.path.join(dist, target), "+w") as f:
        f.write(template.render(render_data))

class Platform(str, Enum):
    linux_armv6:str = "linux.armv6" # Deprecated
    linux_armv7:str = "linux.armv7"
    linux_x64:str = "linux.x86_64"
    linux_aarch64:str = "linux.aarch64"

class Framework(str, Enum):
    tflite:str = "tflite"
    trt:str = "trt"

class PythonVersion(str, Enum):
    py37:str = "py37"
    py38:str = "py38"
    py39:str = "py39"

def copy_model(model_path, package_name, framework, distribution, model_name = 'model') -> str:

    _model = model_name
    if framework == 'tflite':
        _model += '.tflite'
    elif framework == "trt":
        _model += '.engine'

    dest_model_file_path = os.path.join(distribution, package_name, f"libs/{framework}", _model)
    shutil.copyfile(model_path, dest_model_file_path)
    return dest_model_file_path


def _build(framework, platform, pkg_version, pkg_py_version, pkg_name, distribution, is_obf):
    reqs_list = ["opencv-python"]
    if framework == "tflite":
        if platform == "linux.x86_64":
            reqs_list.append("tensorflow")
        elif platform == "linux.armv7":
            if pkg_py_version == "py37":
                reqs_list.append(
                    f"tflite_runtime @ https://github.com/nota-github/tfliteruntime_bin/releases/download/v2.9.0/tflite_runtime-2.9.0rc1-cp{pkg_py_version.split('py')[-1]}-cp{pkg_py_version.split('py')[-1]}m-linux_armv7l.whl"
                )
            else:
                reqs_list.append(
                    f"tflite_runtime @ https://github.com/nota-github/tfliteruntime_bin/releases/download/v2.9.0/tflite_runtime-2.9.0rc1-cp{pkg_py_version.split('py')[-1]}-cp{pkg_py_version.split('py')[-1]}-linux_armv7l.whl"
                )
        elif platform == "linux.aarch64":
            reqs_list.append(
                f"tflite_runtime @ https://github.com/nota-github/tfliteruntime_bin/releases/download/v2.9.0/tflite_runtime-2.9.0rc0-cp{pkg_py_version.split('py')[-1]}-none-linux_aarch64.whl"
            )
    
    packages = [
        f"{pkg_name}",
        f"{pkg_name}.libs",
        f"{pkg_name}.libs.{framework}",
        f"{pkg_name}.models",
        f"{pkg_name}.utils",
    ]
    package_data = {
        f"{pkg_name}.libs.{framework}" : ['*.tflite', '*.trt', '*.engine'],
    }
    platform_dict = {
        "linux.x86_64":"linux.x86_64",
        "linux.armv7":"linux.armv7l",
        "linux.aarch64":"linux.aarch64"
    }

    setup(
        name=pkg_name,
        version=pkg_version,
        description="",
        long_description="",
        platforms="Posix;",
        zip_safe=False,
        python_requires="",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Operating System :: POSIX :: Linux",
            f"Programming Language :: Python :: {pkg_py_version}"
        ],
        include_package_data=True,
        packages=packages,
        package_dir={
            "": os.path.relpath(os.path.join(distribution), os.getcwd())
        },
        package_data=package_data,
        install_requires=reqs_list,
        script_args = [
            'build', 
            '--build-purelib', os.path.join(distribution, pkg_name),
            
            "bdist_wheel",
            "--plat-name", platform_dict[platform],
            "--python-tag", pkg_py_version,
            "--dist-dir", distribution,
            "--bdist-dir", os.path.join(distribution, pkg_name),
        ]
    )

def build_package(data:dict, output_dir:str="dist/package"):

    model_class_names:[str] = data.get("model_class_names")
    model_urls:[str] = data.get("model_urls")
    eingine_file_names:[str] = data.get("eingine_file_names")
    source_path:str = data.get("source_path")
    package_name:str = data.get("package_name")
    platform:str = data.get("platform")
    framework:str = data.get("framework")
    package_version:str = data.get("package_version")
    is_obf:bool = data.get("obf")
    python_version:str = data.get("python_version")
    inference_class_name:str = data.get("inference_class_name")

    # mkdir -> copy -> template -> (obf) -> build & package
    task_id = str(uuid.uuid4())
    logging.info(f'Build start taskID: {task_id}')
    _distribute = os.path.abspath(os.path.join('./dist', ''.join(random.choice(string.ascii_lowercase + '0123456789') for i in range(8))))
    _base = os.path.dirname(__file__)
    logging.info(f'Create package output folder {_distribute}')
    if os.path.exists(_distribute):
        try:
            shutil.rmtree(_distribute)
        except:
            logging.error(f'{_distribute} remove failed')
    os.makedirs(_distribute, exist_ok=False)
    shutil.copytree(os.path.join(_base, "templates"), os.path.join(_distribute, package_name))

    logging.info(f'prepare model for packaging')
    model_path:str = ''
    for i, model_url in enumerate(model_urls):
        tmp_model_path:str = copy_model(
            model_path=model_url,
            package_name=package_name,
            framework=framework,
            distribution=_distribute,
            model_class_names=model_class_names[i],
        )

        if i == 0:
            model_path = tmp_model_path

    template_model_name={
        "tflite":"tflite",
        "trt":"engine"
    }
    inference_template_var = {
        template_model_name[framework]:model_path.split("/")[-1],
        "is_obf":is_obf
    }

    # template
    logging.info(f'Make template')
    env = Environment(loader=FileSystemLoader(os.path.join(_distribute, package_name)))
    make_template(
        env, f"libs/{framework}/inference.py",
        os.path.join(_distribute, package_name),
        inference_template_var
    )

    source = None

    if source_path is not None:
        try:
            source_file = open(source_path, "r")
            source = source_file.read()
        except:
            pass
    
    if source is not None:
        with open(os.path.join(_distribute, package_name, "models/model.py"), "w") as f:
            f.write(source)
    else:
        os.remove(os.path.join(_distribute, package_name, "models/model.py"))

    make_template(
        env, "models/base.py",
        os.path.join(_distribute, package_name),
        {"framework":framework,
         "engine_classes": model_class_names,
         "engine_file_names": eingine_file_names}
    )

    if os.path.exists(os.path.join(_distribute, package_name, "models/model.py")):
        make_template(
            env, "models/model.py", 
            os.path.join(_distribute, package_name),
            {"source":source or "", 
             "model_classes": model_class_names,
             "inference_class_name": inference_class_name}
        )

    # obf
    _obf_ditribute = _distribute
    if is_obf:
        logging.info(f'Obfuscate source codes')
        _obf_ditribute =  os.path.abspath(os.path.join('./dist/obf', ''.join(random.choice(string.ascii_lowercase + '0123456789') for i in range(8))))
        obfuscate(_distribute, _obf_ditribute)

    logging.info(f'Build package')
    _build(
        framework=framework, 
        platform=platform, 
        pkg_py_version=python_version, 
        pkg_version=package_version, 
        pkg_name=package_name, 
        distribution=_obf_ditribute if is_obf else _distribute, 
        is_obf=is_obf
    )

    target_pkg_output_path = os.path.join(output_dir, package_name)
    if os.path.exists(target_pkg_output_path):
        logging.info(f'Remove {target_pkg_output_path}')
        shutil.rmtree(target_pkg_output_path)

    logging.info(f'Copy package files... {target_pkg_output_path}')
    shutil.copytree(_obf_ditribute if is_obf else _distribute, target_pkg_output_path)    
    logging.info(f'Remove {_distribute}')
    if is_obf:
        shutil.rmtree(_obf_ditribute)    
    shutil.rmtree(_distribute)
    return