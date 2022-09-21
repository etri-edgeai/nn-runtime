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

