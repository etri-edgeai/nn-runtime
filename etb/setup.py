from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

SERVER_FLAG = True
if SERVER_FLAG:
    req = [
        "flask",
        "elasticsearch==7.16.3",
        "requests",
        "dockerfile_parse"
    ]
else:
    req = [
        "elasticsearch",
        "requests"
    ]

setup(
    name="etb",
    version="0.1.0",
    author="Jong-Ryul Lee, Eunji Kim, Sungwoo Kim",
    author_email="jongryul.lee@etri.re.kr",
    description="ETB Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    packages=find_packages(),
    install_requires=req
)

