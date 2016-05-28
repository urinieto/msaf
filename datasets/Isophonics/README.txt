This is an example of an MSAF Dataset, which includes the JAM annotations
for the SALAMI and the Isophonics datasets.

You will need access to the audio before running MSAF on this dataset.

Structure:

ds_example/
    audio: In this folder you must place the audio files corresponding to
        the files in the references.
    estimations: MSAF will place the estimations here.
    features: MSAF will place the feature files here.
    references: JAM files for the SALAMI and Isophonics datasets (included).
