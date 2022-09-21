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
import requests

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

class Build(BaseModel):
    framework: Framework
    model_url: Optional[str]
    package_name: Optional[str]
    platform: Platform
    package_version: Optional[str]
    python_version: Optional[PythonVersion]
    requirements: Optional[str]
    obf: Optional[bool]
    upload_url: Optional[str]

    @classmethod
    def as_form(
        cls,
        framework: Framework = Form(...),
        model_path: Optional[str] = Form(None),
        package_name: Optional[str] = Form("edge_ai_runtime"),
        platform: Platform = Form(...), 
        package_version: Optional[str] = Form("0.0.0"),
        python_version: Optional[PythonVersion] = Form("py38"),
        requirements: Optional[str] = Form(""),
        obf: Optional[bool] = Form(True),
        upload_url: Optional[str] = Form(None)
    ):
        return cls(
            framework=framework, 
            model_path=model_path,
            package_name=package_name, 
            platform=platform,
            package_version=package_version, 
            python_version=python_version, 
            requirements=requirements, 
            obf=obf,
            upload_url=upload_url
            )

    @validator("model_path")
    def check_model(cls, model_path):
        if model_path == None:
            return 
        else:
            return model_path

    @validator("platform")
    def check_platform(cls, platform):
        return platform.value

    @validator("framework")
    def check_framework(cls, framework):
        return framework.value

    @validator("python_version")
    def check_python_version(cls, python_version):
        return python_version.value


def copy_model(model_path, package_name, framework, distribution) -> str:

    _model = "model"
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
            '-q',
            'build', 
            '--build-purelib', os.path.join(distribution,"edge_ai"),
            
            "bdist_wheel",
            "--plat-name", platform_dict[platform],
            "--python-tag", pkg_py_version,
            "--dist-dir", distribution,
            "--bdist-dir", os.path.join(distribution,"edge_ai"),
        ]
    )

def build_package(data:dict):
    model_url:str = data.get("model_url")
    source_path:str = data.get("source_path")
    model_file:bytes = data.get("model_file")
    package_name:str = data.get("package_name")
    platform:str = data.get("platform")
    framework:str = data.get("framework")
    package_version:str = data.get("package_version")
    is_obf:bool = data.get("obf")
    python_version:str = data.get("python_version")

    # mkdir -> copy -> template -> (obf) -> build & package
    task_id = str(uuid.uuid4())
    logging.info(f'Build start taskID: {task_id}')
    _distribute = os.path.abspath(os.path.join("/tmp/work", ''.join(random.choice(string.ascii_lowercase + '0123456789') for i in range(8))))
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
    model_path:str = copy_model(
        model_url=model_url,
        model_file=model_file,
        package_name=package_name,
        framework=framework,
        distribution=_distribute
    )

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
    if source is not None:
        with open(os.path.join(_distribute, package_name, "models/model.py"), "w") as f:
            f.write(source)
    else:
        os.remove(os.path.join(_distribute, package_name, "models/model.py"))
    source = ""
    if source_path is not None:
        source_file = open(source_path, "r")
        source = source_file.read()

    make_template(
        env, "models/model.py", 
        os.path.join(_distribute, package_name),
        {"source":source or ""}
    )
    make_template(
        env, "models/base.py",
        os.path.join(_distribute, package_name),
        {"framework":framework}
    )
    
    # obf
    if is_obf:
        logging.info(f'Obfuscate package')
        #_distribute = obfuscate(platform, python_version, _distribute, package_name, framework, model_files)

    logging.info(f'Build package')
    _build(
        framework=framework, 
        platform=platform, 
        pkg_py_version=python_version, 
        pkg_version=package_version, 
        pkg_name=package_name, 
        distribution=_distribute, 
        is_obf=is_obf
    )
    
    # upload
    whl_file = [i for i in os.listdir(_distribute) if os.path.splitext(i)[1] == '.whl'][0]
    logging.info(f'Upload `whl` file {whl_file}')
    
    logging.info(f'Remove {_distribute}')
    if is_obf:
        _distribute = os.path.dirname(_distribute)
    shutil.rmtree(_distribute)
    return