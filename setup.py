from setuptools import setup, find_packages
import glob
import imp
import numpy.distutils.misc_util


version = imp.load_source('msaf.version', 'msaf/version.py')

# MSAF configuration
setup(
    name='msaf',
    version=version.version,
    description='Python module to discover the structure of music files',
    author='Oriol Nieto',
    author_email='oriol@nyu.edu',
    url='https://github.com/urinieto/msaf',
    download_url='https://github.com/urinieto/msaf/releases',
    packages=find_packages(),
    package_data={'msaf': ['algorithms/olda/models/*.npy']},
    data_files=[('msaf/algorithms/olda/models',
                 glob.glob('msaf/algorithms/olda/models/*.npy'))],
    long_description="""A python module to segment audio into all its """
    """different large-scale sections and label them based on their """
    """acoustic similarity""",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4"
    ],
    keywords='audio music sound',
    license='MIT',
    install_requires=[
        'audioread',
        'enum34',
        'future',
        'jams',
        'numpy >= 1.8.0',
        'scipy >= 0.13.0',
        'scikit-learn >= 0.17.0',
        'seaborn',  # For notebook example (but everyone should have this :-))
        'matplotlib',
        'joblib',
        'decorator',
        'cvxopt',
        'joblib',
        'librosa >= 0.4.2',
        'mir_eval',
        'pandas'
    ],
    extras_require={
        'resample': 'scikits.samplerate>=0.3'
    },
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs()
)
