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

int readJSON(const char* json_path) {

    std::filebuf fb;
    fb.open(json_path, ios::in);
    istream inputs(&fb);

    Json::Value root;   // will contains the root value after parsing.
    Json::Reader reader;
    bool parsingSuccessful = reader.parse( inputs, root );
    if ( !parsingSuccessful )
    {
        // report to the user the failure and their locations in the document.
        std::cout  << "Failed to parse configuration\n"
                   << reader.getFormattedErrorMessages();
        return 1;
    }

    cout << root << endl;
    fb.close();

    return 0;
}

int main(int argc, char const *argv[])
{
    printf("Initialising...\n");
    ClusterMeltSegmenterParams params; // use default params
    ClusterMeltSegmenter *segmenter = new ClusterMeltSegmenter(params);

    // Open JSON
    readJSON("/Users/uri/datasets/Segments/estimations/Isophonics_01_-_No_Reply.json");

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