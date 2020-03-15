#!/usr/bin/env python

import os
from setuptools import setup, find_packages
import sys
import platform
import glob

def get_install_requirements(path):
    content = open(os.path.join(os.path.dirname(__file__), path)).read()
    return [
        req
        for req in content.split("\n")
        if req != '' and not req.startswith('#')
    ]

def get_version(path):
    content = open(os.path.join(os.path.dirname(__file__), path)).read()
    return content

setup(name='eva4',
        version=get_version("VERSION"),
        description='EVA4',
        url='https://github.com/hemantkashniyal/EVA4',
        author='Hemant Kashniyal',
        author_email='hemantkashniyal@gmail.com',
        license='Proprietary',
        packages=['eva4'],
        package_data={'': ['requirements.txt', 'VERSION']},
        include_package_data=True,
        zip_safe=False,
        install_requires=get_install_requirements("requirements.txt")
        )
