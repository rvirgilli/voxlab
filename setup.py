# setup.py

from setuptools import setup, find_packages

setup(
    name='voxlab',
    version='0.1',
    packages=find_packages(),
    py_modules=['voxlab'],
    install_requires=[
        # List your dependencies here, for example:
        # 'numpy',
        # 'scipy',
    ],
    entry_points={
        'console_scripts': [
            # Define any command line scripts here
        ],
    },
    author='Rafaello Virgilli',
    author_email='rvirgilli@gmail.com',
    description='A toolbox for audio processing and voice deep learning models.',
    url='https://github.com/rvirgilli/voxlab',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
