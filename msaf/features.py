"""
MSAF module with the available features.

Each feature must inherit from the base class `Features` to be
included in the whole framework.
"""

from builtins import super
import librosa
import numpy as np

# Local stuff
import msaf
from msaf import utils
from msaf import input_output as io
from msaf.input_output import FileStruct
from msaf.base import Features


class CQT(Features):
    """This class contains the implementation of the Constant-Q Transform.

    These features contain both harmonic and timbral content of the given
    audio signal.
    """
    def __init__(self, file_struct, feat_type, sr=msaf.Anal.sample_rate,
                 hop_length=msaf.Anal.hop_size, n_bins=msaf.Anal.cqt_bins,
                 norm=msaf.Anal.cqt_norm,
                 filter_scale=msaf.Anal.cqt_filter_scale, ref_power=np.max):
        """Constructor of the class.

        Parameters
        ----------
        file_struct: `msaf.input_output.FileStruct`
            Object containing the file paths from where to extract/read
            the features.
        feat_type: `FeatureTypes`
            Enum containing the type of features.
        sr: int > 0
            Sampling rate for the analysis.
        hop_length: int > 0
            Hop size in frames for the analysis.
        n_bins: int > 0
            Number of frequency bins for the CQT.
        norm: float
            The normalization coefficient.
        filter_scale: float
            The scale of the filter for the CQT.
        ref_power: function
            The reference power for logarithmic scaling.
        """
        # Init the parent
        super().__init__(file_struct=file_struct, sr=sr, hop_length=hop_length,
                         feat_type=feat_type)
        # Init the CQT parameters
        self.n_bins = n_bins
        self.norm = norm
        self.filter_scale = filter_scale
        self.ref_power = ref_power

    def get_id(self):
        """Identifier of these features."""
        return "cqt"

    def compute_features(self):
        """Actual implementation of the features.

        Returns
        -------
        cqt: np.array(N, F)
            The features, each row representing a feature vector for a give
            time frame/beat.
        """
        linear_cqt = np.abs(librosa.cqt(
            self._audio, sr=self.sr, hop_length=self.hop_length,
            n_bins=self.n_bins, norm=self.norm, filter_scale=self.filter_scale,
            real=False)) ** 2
        cqt = librosa.logamplitude(linear_cqt, ref_power=self.ref_power).T
        return cqt


class MFCC(Features):
    """This class contains the implementation of the MFCC Features.

    The Mel-Frequency Cepstral Coefficients contain timbral content of a
    given audio signal.
    """
    def __init__(self, file_struct, feat_type, sr=msaf.Anal.sample_rate,
                 hop_length=msaf.Anal.hop_size, n_fft=msaf.Anal.n_fft,
                 n_mels=msaf.Anal.n_mels, n_mfcc=msaf.Anal.n_mfcc,
                 ref_power=np.max):
        """Constructor of the class.

        Parameters
        ----------
        file_struct: `msaf.input_output.FileStruct`
            Object containing the file paths from where to extract/read
            the features.
        feat_type: `FeatureTypes`
            Enum containing the type of features.
        sr: int > 0
            Sampling rate for the analysis.
        hop_length: int > 0
            Hop size in frames for the analysis.
        n_fft: int > 0
            Number of frames for the FFT.
        n_mels: int > 0
            Number of mel filters.
        n_mfcc: int > 0
            Number of mel coefficients.
        ref_power: function
            The reference power for logarithmic scaling.
        """
        # Init the parent
        super().__init__(file_struct=file_struct, sr=sr, hop_length=hop_length,
                         feat_type=feat_type)
        # Init the MFCC parameters
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.ref_power = ref_power

    def get_id(self):
        """Identifier of these features."""
        return "mfcc"

    def compute_features(self):
        """Actual implementation of the features.

        Returns
        -------
        mfcc: np.array(N, F)
            The features, each row representing a feature vector for a give
            time frame/beat.
        """
        S = librosa.feature.melspectrogram(self._audio,
                                           sr=self.sr,
                                           n_fft=self.n_fft,
                                           hop_length=self.hop_length,
                                           n_mels=self.n_mels)
        log_S = librosa.logamplitude(S, ref_power=self.ref_power)
        mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=self.n_mfcc).T
        return mfcc


#def compute_features(audio, y_harmonic):
    #"""Computes the HPCP and MFCC features.

    #Parameters
    #----------
    #audio: np.array(N)
        #Audio samples of the given input.
    #y_harmonic: np.array(N)
        #Harmonic part of the audio signal, in samples.

    #Returns
    #-------
    #mfcc: np.array(N, msaf.Anal.mfcc_coeff)
        #Mel-frequency Cepstral Coefficients.
    #pcp: np.array(N, 12)
        #Pitch Class Profiles.
    #tonnetz: np.array(N, 6)
        #Tonal Centroid features.
    #cqt: np.array(N, msaf.Anal.cqt_bins)
        #Constant-Q log-scale features.
    #tempogram: np.array(N, 192)
        #Tempogram features.
    #"""
    #logging.info("Computing Spectrogram...")
    ##S = librosa.feature.melspectrogram(audio,
                                       ##sr=msaf.Anal.sample_rate,
                                       ##n_fft=msaf.Anal.frame_size,
                                       ##hop_length=msaf.Anal.hop_size,
                                       ##n_mels=msaf.Anal.n_mels)

    #logging.info("Computing Constant-Q...")
    ##cqt = librosa.logamplitude(librosa.cqt(audio, sr=msaf.Anal.sample_rate,
                                           ##hop_length=msaf.Anal.hop_size,
                                           ##n_bins=msaf.Anal.cqt_bins) ** 2,
                               ##ref_power=np.max).T

    ##linear_cqt = np.abs(librosa.cqt(y_harmonic,
                                    ##sr=msaf.Anal.sample_rate,
                                    ##hop_length=msaf.Anal.hop_size,
                                    ##n_bins=msaf.Anal.cqt_bins,
                                    ##norm=np.inf,
                                    ##filter_scale=1,
                                    ##real=False))
    ##cqt = librosa.logamplitude(linear_cqt, ref_power=np.max).T

    #logging.info("Computing MFCCs...")
    ##log_S = librosa.logamplitude(S, ref_power=np.max)
    ##mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=msaf.Anal.mfcc_coeff).T

    #logging.info("Computing PCPs...")
    #pcp_cqt = np.abs(librosa.hybrid_cqt(y_harmonic,
                                        #sr=msaf.Anal.sample_rate,
                                        #hop_length=msaf.Anal.hop_size,
                                        #n_bins=msaf.Anal.cqt_bins,
                                        #norm=1,
                                        #fmin=f_min)) ** 2
    #pcp = librosa.feature.chroma_cqt(C=pcp_cqt,
                                     #sr=msaf.Anal.sample_rate,
                                     #hop_length=msaf.Anal.hop_size,
                                     #n_octaves=msaf.Anal.n_octaves,
                                     #fmin=f_min).T
    ##pcp = librosa.feature.chroma_cqt(C=linear_cqt,
                                     ##sr=msaf.Anal.sample_rate,
                                     ##hop_length=msaf.Anal.hop_size,
                                     ##n_octaves=msaf.Anal.n_octaves,
                                     ##fmin=msaf.Anal.f_min).T
    ##pcp = librosa.feature.chroma_stft(y=y_harmonic,
                                      ##sr=msaf.Anal.sample_rate,
                                      ##n_fft=msaf.Anal.frame_size,
                                      ##hop_length=msaf.Anal.hop_size).T

    #logging.info("Computing Tonnetz...")
    ##tonnetz = utils.chroma_to_tonnetz(pcp)

    ## TODO:
    #tonnetz = pcp
    #mfcc = pcp
    #cqt = pcp

    #logging.info("Computing Tempogram...")
    #tempogram = librosa.feature.tempogram(audio,
                                          #sr=msaf.Anal.sample_rate,
                                          #hop_length=msaf.Anal.hop_size,
                                          #win_length=192).T
    #return mfcc, pcp, tonnetz, cqt, tempogram




#def compute_beat_sync_features(features, beats_idx):
    #"""Given a dictionary of features, and the estimated index frames,
    #calculate beat-synchronous features."""
    #pad = True  # Always pad til the end of the actual audio file
    #bs_mfcc = librosa.feature.sync(features["mfcc"].T, beats_idx, pad=pad).T
    #bs_pcp = librosa.feature.sync(features["pcp"].T, beats_idx, pad=pad).T
    #bs_tonnetz = librosa.feature.sync(features["tonnetz"].T, beats_idx,
                                      #pad=pad).T
    #bs_cqt = librosa.feature.sync(features["cqt"].T, beats_idx, pad=pad).T
    #bs_tempogram = librosa.feature.sync(features["tempogram"].T, beats_idx,
                                        #pad=pad).T

    ## Make sure we have the right size (remove last frame if needed)
    #bs_mfcc = bs_mfcc[:len(beats_idx), :]
    #bs_pcp = bs_pcp[:len(beats_idx), :]
    #bs_tonnetz = bs_tonnetz[:len(beats_idx), :]
    #bs_cqt = bs_cqt[:len(beats_idx), :]
    #bs_tempogram = bs_tempogram[:len(beats_idx), :]
    #return bs_mfcc, bs_pcp, bs_tonnetz, bs_cqt, bs_tempogram
