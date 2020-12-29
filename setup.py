import os
from setuptools import setup, find_packages

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))

REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, 'requirements.txt')

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

install_reqs.append('setuptools')

setup(
    name='Part2ProjectUpdated',
    version='0.1.0',
    author='Melissa Yao',
    author_email='myao1135@gmail.com',
    packages=find_packages(),
    package_data={'': ['LICENSE.txt',
                       'README.md',
                       'requirements.txt']
                  },
    url='https://github.com/myao20/Part2ProjectUpdated/',
    license='MIT',
    description='Part 2 Project Code',
    long_description=open('README.md').read(),
    install_requires=install_reqs
)
