import argparse

import msaf

import segmenter

# Inject custom segmenter.
setattr(msaf.algorithms, segmenter.algo_id, segmenter)
msaf.algorithms.__all__.append(segmenter.algo_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path", help="audio file")
    args = parser.parse_args()
    results = msaf.run.process(args.in_path, boundaries_id=segmenter.algo_id)
    print(results)


if __name__ == "__main__":
    main()
