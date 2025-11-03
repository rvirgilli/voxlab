from setuptools import setup, find_packages

# Read requirements from the file
with open('voxlab/requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='voxlab',
    version='0.3.7',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            
        ],
    },
    author='Rafaello Virgilli',
    author_email='rvirgilli@gmail.com',
    description='A comprehensive Python toolbox for audio processing using PyTorch with device-aware operations and GPU acceleration.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/rvirgilli/voxlab',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    keywords='audio processing, pytorch, torchaudio, signal processing, gpu acceleration',
    python_requires='>=3.11',
)