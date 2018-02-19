import re
import os

from setuptools import setup, find_packages


def get_version():
    filepath = os.path.join(
        os.path.dirname(__file__), 'cnn_finetune', '__init__.py'
    )
    with open(filepath) as f:
        return re.findall("__version__ = '([\d.\w]+)'", f.read())[0]


def get_long_description():
    with open('README.md') as f:
        return f.read()


requirements = [
    'torch',
    'torchvision',
    'pretrainedmodels',
    'scipy', # required for torchvision
    'tqdm',  # required for pretrainedmodels
]


tests_requirements = [
    'pytest',
    'numpy',
]


setup(
    name='cnn_finetune',
    version=get_version(),
    description=(
        'Fine-tune pretrained Convolutional Neural Networks with PyTorch'
    ),
    long_description=get_long_description(),
    author='Alex Parinov',
    author_email='creafz@gmail.com',
    url='https://github.com/creafz/pytorch-cnn-finetune',
    license='MIT',
    install_requires=requirements,
    extras_require={'tests': tests_requirements},
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(exclude=['tests', 'examples']),
)
