from setuptools import setup, Extension
import sys
import numpy.distutils.misc_util

extra_compile_flags = ""
extra_linker_flags = ""
if "linux" in sys.platform:
    extra_compile_flags = "-std=c++11 -DUSE_PTHREADS"
elif "darwin" in sys.platform:
    extra_compile_flags = "-DUSE_PTHREADS"
    extra_linker_flags = "-framework Accelerate"

cc_segmenter = Extension("cc_segmenter",
                         sources=["base/Pitch.cpp",
                                  "dsp/chromagram/Chromagram.cpp",
                                  "dsp/chromagram/ConstantQ.cpp",
                                  "dsp/keydetection/GetKeyMode.cpp",
                                  "dsp/mfcc/MFCC.cpp",
                                  "dsp/onsets/DetectionFunction.cpp",
                                  "dsp/onsets/PeakPicking.cpp",
                                  "dsp/phasevocoder/PhaseVocoder.cpp",
                                  "dsp/rateconversion/Decimator.cpp",
                                  "dsp/rhythm/BeatSpectrum.cpp",
                                  "dsp/segmentation/cluster_melt.c",
                                  "dsp/segmentation/ClusterMeltSegmenter.cpp",
                                  "dsp/segmentation/cluster_segmenter.c",
                                  "dsp/segmentation/Segmenter.cpp",
                                  "dsp/signalconditioning/DFProcess.cpp",
                                  "dsp/signalconditioning/Filter.cpp",
                                  "dsp/signalconditioning/FiltFilt.cpp",
                                  "dsp/signalconditioning/Framer.cpp",
                                  "dsp/tempotracking/DownBeat.cpp",
                                  "dsp/tempotracking/TempoTrack.cpp",
                                  "dsp/tempotracking/TempoTrackV2.cpp",
                                  "dsp/tonal/ChangeDetectionFunction.cpp",
                                  "dsp/tonal/TCSgram.cpp",
                                  "dsp/tonal/TonalEstimator.cpp",
                                  "dsp/transforms/FFT.cpp",
                                  "dsp/wavelet/Wavelet.cpp",
                                  "hmm/hmm.c",
                                  "maths/Correlation.cpp",
                                  "maths/CosineDistance.cpp",
                                  "maths/KLDivergence.cpp",
                                  "maths/MathUtilities.cpp",
                                  "maths/pca/pca.c",
                                  "thread/Thread.cpp",
                                  "main.cpp"
                                  ],
                         include_dirs=["dsp/segmentation",
                                       ".",
                                       "include"],
                         libraries=["jsoncpp",
                                    "stdc++"],
                         extra_compile_args=[extra_compile_flags],
                         extra_link_args=[extra_linker_flags],
                         language="c++")

setup(ext_modules=[cc_segmenter],
      include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs())
