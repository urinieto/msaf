from setuptools import setup, find_packages

setup(
    name='msaf',
    version='0.0.2',
    description='Python module to discover the structure of music files',
    author='Oriol Nieto',
    author_email='oriol@nyu.edu',
    url='https://github.com/urinieto/msaf',
    download_url='https://github.com/urinieto/msaf/releases',
    #packages=['msaf'],
    packages=find_packages(),
    #package_data={'': ['ds_example/*']},
    long_description="""A python module to segment audio into all its """
        """different large-scale sections and label them based on their """
        """acoustic similarity""",
    classifiers=[
        "License :: OSI Approved :: GPL 3",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7"
        ],
    keywords='audio music sound',
    license='GPL',
    install_requires=[
        'audioread',
        'numpy >= 1.8.0',
        'scipy >= 0.13.0',
        'scikit-learn >= 0.14.0',
        'matplotlib',
        'joblib',
        'decorator',
        'cvxopt',
        'joblib',
        'librosa',
        'mir_eval',
        'pandas'
    ],
    extras_require={
        'resample': 'scikits.samplerate>=0.3'
        }
)
