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
#include <json/json.h>

using namespace std;

/******** READ AND WRITE JSON FILES *********/

/*
Transform JSON Value to a 2D Vector (vector<vector<double> >)
*/
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

/*
Transform JSON Value to a 1D Vector (vector<double>)
*/
vector<double> JSONValueToVector(Json::Value &data) {

    int T = data.size();

    vector<double> f;
    f.resize(T);

    for (int i = 0; i < T; ++i)
    {
        f[i] = data[i].asDouble();
    }

    // Make unique
    vector<double>::iterator it;
    it = unique (f.begin(), f.end());
    f.resize( std::distance(f.begin(),it) );

    return f;
}

vector<double> JAMSValueToVector(Json::Value &data) {

    int T = data.size();

    vector<double> f;
    f.resize(T);

    for (int i = 0; i < T; ++i)
    {
        f[i] = data[i]["time"]["value"].asDouble();
    }

    // Make unique
    vector<double>::iterator it;
    it = unique (f.begin(), f.end());
    f.resize( std::distance(f.begin(),it) );

    // for (int i = 0; i < f.size(); ++i)
    // {
    //     cout << f[i] << endl;
    // }

    return f;
}

vector<vector<double> > readJSON(const char* json_path, bool annot_beats,
                        const char* feature) {

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

    if (strcmp(feature, "beats") == 0) {
        f.resize(1);
        f[0] = JSONValueToVector(root[feature]["ticks"][0]);
    }
    else {
        char key[64];
        if (annot_beats)
            sprintf(key, "ann_beatsync");
        else
            sprintf(key, "est_beatsync");

        f = JSONValueTo2DVector(root[key][feature]);

        fb.close();
    }

    return f;
}

vector<double> readBeatsJAMS(const char *jams_path) {

    std::filebuf fb;
    fb.open(jams_path, ios::in);
    istream inputs(&fb);

    Json::Value root;   // will contains the root value after parsing.
    Json::Reader reader;
    bool parsingSuccessful = reader.parse( inputs, root );

    vector<double> beats;
    if ( !parsingSuccessful )
    {
        // report to the user the failure and their locations in the document.
        std::cout  << "Failed to parse configuration\n"
                   << reader.getFormattedErrorMessages();
        return beats;
    }

    // cout << root["beats"][0]["data"] << endl;
    beats = JAMSValueToVector(root["beats"][0]["data"]);

    return beats;
}

void createEstimation(Json::Value &est, vector<double> &times, 
        bool annot_beats, const char* feature) {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y/%m/%d %H:%M:%S", &tstruct);

    Json::Value vec(Json::arrayValue);

    est["annot_beats"] = annot_beats;
    est["timestamp"] = buf;

    for (int i = 0; i < times.size(); ++i)
    {
        vec.append(Json::Value(times[i]));
    }
    est["data"] = vec;
    est["feature"] = feature;
    est["version"] = "1.0";

}

void writeJSON(const char* est_file, vector<double> times, bool annot_beats, 
               const char* feature) {

    cout << "Writing to: " << est_file << endl;

    std::filebuf fb;
    fb.open(est_file, ios::in);
    istream inputs(&fb);

    Json::Value root;   // will contains the root value after parsing.
    Json::Reader reader;
    bool parsingSuccessful = reader.parse( inputs, root );

    vector<double> beats;
    if ( !parsingSuccessful )
    {
        // report to the user the failure and their locations in the document.
        std::cout  << "Failed to parse configuration\n"
                   << reader.getFormattedErrorMessages();
        return;
    }

    Json::Value est;
    createEstimation(est, times, annot_beats, feature);

    if (root["boundaries"]["levy"] == Json::nullValue) {
        Json::Value vec(Json::arrayValue);
        vec.append(est);
        root["boundaries"]["levy"] = vec;
    }
    else {
        // Find possible value to overwrite
        bool found = false;
        for (int i = 0; i < root["boundaries"]["levy"].size(); ++i)
        {
            if (strcmp(root["boundaries"]["levy"][i]["feature"].asCString(), feature) == 0
                && root["boundaries"]["levy"][i]["annot_beats"].asBool() == annot_beats) {

                found = true;
                root["boundaries"]["levy"][i] = est;
                break;
            }
        }
        if (!found) {
            root["boundaries"]["levy"].append(est);
        }
    }

    // cout << root << endl;

    // Write JSON
    ofstream myfile;
    myfile.open (est_file);
    myfile << root;
    myfile.close();
}

void writeResults(Segmentation segmentation, bool annot_beats,
                  const char* feats_path, const char *feature) {

    vector<double> beats;
    string feat_path = feats_path;

    // If annotated beats, read the beats from the JAMS file
    if (annot_beats) {
        // Get jams file
        size_t idx = feat_path.find("features");
        string annot_path = feat_path.replace(idx, 8, "annotations");
        idx = annot_path.find(".json");
        string jams_path = annot_path.replace(idx, 5, ".jams");

        // Read Beats
        beats = readBeatsJAMS(jams_path.c_str());
    }
    // Else, read the beats from the features JSON file
    else {
        // Get beats from JSON
        vector<vector<double> > beats_placeholder;
        beats_placeholder = readJSON(feats_path, annot_beats, "beats");
        beats.resize(beats_placeholder[0].size());
        for (int i = 0; i < beats.size(); ++i)
        {
            beats[i] = beats_placeholder[0][i];
        }
    }

    // Get segmentation times 
    vector<double> times;
    times.resize(segmentation.segments.size() + 1);
    times[0] = beats[segmentation.segments[0].start];
    for (int i = 0; i < segmentation.segments.size(); ++i)
    {
        times[i+1] = beats[segmentation.segments[i].end];
    }

    // Get estimation file
    string est_path;
    if (annot_beats) {
        size_t idx = feat_path.find("annotations");
        est_path = feat_path.replace(idx, 11, "estimations");
    }
    else {
        size_t idx = feat_path.find("features");
        est_path = feat_path.replace(idx, 8, "estimations");
    }

    // Write results
    writeJSON(est_path.c_str(), times, annot_beats, feature);
    
}

int main(int argc, char const *argv[])
{

    const char *feature = argv[3]; // mffc or hpcp

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
    vector<vector<double> > f = readJSON(argv[1], atoi(argv[2]), feature);

    // Segment until we have a potentially good result
    Segmentation segmentation;
    do {
        // Initialize segmenter
        segmenter->initialise(11025);

        // Set features
        segmenter->setFeatures(f);

        // Segment
        segmenter->segment();
        segmentation = segmenter->getSegmentation();
    } while(segmentation.segments.size() <= 2 && f.size() >= 90);

    // Write the results
    writeResults(segmentation, atoi(argv[2]), argv[1], feature);

    // Clean up
    delete segmenter;

    // Done
    return 0;
}