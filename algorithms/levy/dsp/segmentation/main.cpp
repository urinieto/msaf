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
#include <vector>

using namespace std;

int main(int argc, char const *argv[])
{
    printf("Initialising...\n");
    ClusterMeltSegmenterParams params; // use default params
    ClusterMeltSegmenter *segmenter = new ClusterMeltSegmenter(params);

    // Read features from JAMS
    vector<vector<double> > f;

    // Set features
    segmenter->setFeatures(f);

    // Segment
    segmenter->segment();

    // Clean up
    delete segmenter;

    // Done
    return 0;
}