from setuptools import setup, Extension, find_packages
import glob
import sys
import numpy.distutils.misc_util
from distutils import sysconfig

# Compile the CC algorithm
extra_compile_flags = ""
extra_linker_flags = ""
if "linux" in sys.platform:
    extra_compile_flags = "-std=c++11 -DUSE_PTHREADS"
    extra_linker_flags = "-llapack -lblas -lm"
elif "darwin" in sys.platform:
    extra_compile_flags = "-DUSE_PTHREADS"
    extra_linker_flags = "-framework Accelerate"

cc_path = "msaf/algorithms/cc/"
cc_segmenter = Extension(cc_path + "cc_segmenter",
                         sources=[cc_path + "base/Pitch.cpp",
                                  cc_path + "dsp/chromagram/Chromagram.cpp",
                                  cc_path + "dsp/chromagram/ConstantQ.cpp",
                                  cc_path + "dsp/mfcc/MFCC.cpp",
                                  cc_path + "dsp/onsets/DetectionFunction.cpp",
                                  cc_path + "dsp/phasevocoder/PhaseVocoder.cpp",
                                  cc_path + "dsp/rateconversion/Decimator.cpp",
                                  cc_path + "dsp/segmentation/cluster_melt.c",
                                  cc_path + "dsp/segmentation/ClusterMeltSegmenter.cpp",
                                  cc_path + "dsp/segmentation/cluster_segmenter.c",
                                  cc_path + "dsp/segmentation/Segmenter.cpp",
                                  cc_path + "dsp/transforms/FFT.cpp",
                                  cc_path + "hmm/hmm.c",
                                  cc_path + "maths/Correlation.cpp",
                                  cc_path + "maths/CosineDistance.cpp",
                                  cc_path + "maths/MathUtilities.cpp",
                                  cc_path + "maths/pca/pca.c",
                                  cc_path + "main.cpp"
                                  ],
                         include_dirs=[cc_path + "dsp/segmentation",
                                       cc_path,
                                       cc_path + "include"],
                         libraries=["stdc++"],
                         extra_compile_args=[extra_compile_flags],
                         extra_link_args=[extra_linker_flags],
                         language="c++")

# MSAF configuration
setup(
    name='msaf',
    version='0.0.2',
    description='Python module to discover the structure of music files',
    author='Oriol Nieto',
    author_email='oriol@nyu.edu',
    url='https://github.com/urinieto/msaf',
    download_url='https://github.com/urinieto/msaf/releases',
    packages=find_packages(),
    #data_files=[('msaf/algorithms/olda/models', glob.glob('msaf/algorithms/olda/models/*.npy')),
                #('msaf/algorithms/cc', ['co_segmenter.so'])],
    data_files=[('msaf/algorithms/olda/models',
                 glob.glob('msaf/algorithms/olda/models/*.npy'))],
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
        'librosa >= 0.4.0rc1',
        'mir_eval',
        'pandas'
    ],
    extras_require={
        'resample': 'scikits.samplerate>=0.3'
    },
    ext_modules=[cc_segmenter],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs()
)
