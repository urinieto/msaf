import argparse
import librosa
import msaf


class MyFeatures(msaf.features.Features):
    def __init__(
        self,
        file_struct,
        feat_type,
        sr=msaf.config.sample_rate,
        hop_length=msaf.config.hop_size,
    ):
        super().__init__(
            file_struct=file_struct, sr=sr, hop_length=hop_length, feat_type=feat_type
        )

    @classmethod
    def get_id(cls):
        return "my_features"

    def compute_features(self):
        features = librosa.feature.chroma_vqt(
            y=self._audio,
            sr=self.sr,
            hop_length=self.hop_length,
            intervals="pythagorean",
        )
        features = features.T  # (num_frames, num_features)
        return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path", help="audio file")
    args = parser.parse_args()
    results = msaf.run.process(args.in_path, feature="my_features")
    print(results)


if __name__ == "__main__":
    main()
