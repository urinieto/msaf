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
#include <json/json.h>

using namespace std;

vector<vector<double> > JSONValueTo2DVector(Json::Value &data) {

    int T = data.size();
    int F = data[0].size();

    vector<vector<double> > f;
    f.resize(T);

    for (int i = 0; i < T; ++i)
    {
        f[i].resize(F);
        for (int j = 0; j < F; ++j)
        {
            f[i][j] = data[i][j].asDouble();
        }
    }

    return f;
}

vector<vector<double> > readJSON(const char* json_path, bool annot_beats) {

    std::filebuf fb;
    fb.open(json_path, ios::in);
    istream inputs(&fb);

    Json::Value root;   // will contains the root value after parsing.
    Json::Reader reader;
    bool parsingSuccessful = reader.parse( inputs, root );

    vector<vector<double> > f;
    if ( !parsingSuccessful )
    {
        // report to the user the failure and their locations in the document.
        std::cout  << "Failed to parse configuration\n"
                   << reader.getFormattedErrorMessages();
        return f;
    }

    char key[64];
    if (annot_beats)
        sprintf(key, "ann_beatsync");
    else
        sprintf(key, "est_beatsync");

    f = JSONValueTo2DVector(root[key]["mfcc"]);

    fb.close();

    return f;
}

int main(int argc, char const *argv[])
{

    ClusterMeltSegmenterParams params;
    params.featureType = FEATURE_TYPE_MFCC;
    // params.featureType = FEATURE_TYPE_CHROMA;
    ClusterMeltSegmenter *segmenter = new ClusterMeltSegmenter(params);

    // Read features from JAMS
    vector<vector<double> > f = readJSON(argv[1], atoi(argv[2]));

    // Initialize segmenter
    segmenter->initialise(11025);

    // Set features
    segmenter->setFeatures(f);

    // Segment
    segmenter->segment();

    // Get segmentation
    Segmentation segmentation = segmenter->getSegmentation();

    for (int i = 0; i < segmentation.segments.size(); ++i)
    {
        cout << segmentation.segments[i].start << "\t, " << 
            segmentation.segments[i].end << "\t: " << 
            segmentation.segments[i].type << endl;
    }

    // Clean up
    delete segmenter;

    // Done
    return 0;
}