/*
Little program to run the segmenter using the MSAF features

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"
*/

#include "ClusterMeltSegmenter.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <time.h>
#include "json_io.h"

using namespace std;


int main(int argc, char const *argv[])
{

    if ( argc != 5 ) {
        cout << "Usage:\n\t" << argv[0] << " path_to_features(file.json) annot_beats(0 or 1) features(mfcc|hpcp) annot_bounds(0 or 1)" << endl;
        return -1;
    }
    const char *feature = argv[3]; // mffc or hpcp

    Segmentation segmentation;
    int nframes = getFileFrames(argv[1]);
    bool annot_beats = atoi(argv[2]);
    bool annot_bounds = atoi(argv[4]);

    /* Only segment if duration is greater than 15 seconds */
    if (nframes * 2048 > 15 * 11025) {
        ClusterMeltSegmenterParams params;

        if (strcmp(feature, "mfcc")) {
            params.featureType = FEATURE_TYPE_MFCC;
        }
        else if (strcmp(feature, "hpcp")) {
            params.featureType = FEATURE_TYPE_CHROMA;
        }
        // Set original paper parameters
        params.nHMMStates = 80;
        params.nclusters = 6;
        params.neighbourhoodLimit = 16;

        ClusterMeltSegmenter *segmenter = new ClusterMeltSegmenter(params);

        // Read features from JAMS
        vector<vector<double> > f = readJSON(argv[1], annot_beats, feature);

        // Segment until we have a potentially good result
        do {
            // Initialize segmenter
            segmenter->initialise(11025);

            // Set features
            segmenter->setFeatures(f);

            // Set Annotated Boundaries Indeces if needed
            if (annot_bounds) {
                vector<int> annotBounds = getAnnotBoundIdxs();
                segmenter->setAnnotBounds(annotBounds);
            }

            // Segment
            segmenter->segment();
            segmentation = segmenter->getSegmentation();
        } while(segmentation.segments.size() < 2 && f.size() >= 90);

        // Clean up
        delete segmenter;
    }
    else {
        Segment s;
        s.start = 0;
        s.end = nframes;
        s.type = 0;
        segmentation.segments.push_back(s);
    }

    //cout << segmentation.segments.size() << endl;
    cout << "estimated labels: ";
    for (auto b : segmentation.segments) {
        cout << b.type << " ";
    }
    cout << endl;

    // Write the results
    writeResults(segmentation, annot_beats, annot_bounds, argv[1], feature);

    // Done
    return 0;
}
