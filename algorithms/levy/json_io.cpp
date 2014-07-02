/** Some functions to read and write JSON files.
 *
 * author: Oriol Nieto, 2014
 * **/

#include "json_io.h"

/*
Transform JSON Value to a 2D Vector (vector<vector<double> >)
*/
vector<vector<double> > JSONValueTo2DVector(Json::Value &data) {

    int T = data.size();
    int F = data[0].size();

    vector<vector<double> > f;
    f.resize(T);

    for (int i = 0; i < T; ++i) {
        f[i].resize(F);
        for (int j = 0; j < F; ++j) {
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

    for (int i = 0; i < T; ++i) {
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

    for (int i = 0; i < T; ++i) {
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
    if ( !parsingSuccessful ) {
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

    Json::Value root;   // will contain the root value after parsing.
    Json::Reader reader;
    bool parsingSuccessful = reader.parse( inputs, root );

    vector<double> beats;
    if ( !parsingSuccessful )
    {
        // report to the user the failure and their locations in the document.
        std::cout << "Failed to parse configuration\n"
                  << reader.getFormattedErrorMessages();
        return beats;
    }

    // cout << root["beats"][0]["data"] << endl;
    beats = JAMSValueToVector(root["beats"][0]["data"]);

    return beats;
}

string getJAMSPath(string feat_path) {
    size_t idx = feat_path.find("features");
    string annot_path = feat_path.replace(idx, 8, "annotations");
    idx = annot_path.find(".json");
    string jams_path = annot_path.replace(idx, 5, ".jams");
    return jams_path;
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

void createEstimationInt(Json::Value &est, vector<int> &labels, 
        bool annot_beats, bool annot_bounds, const char* feature) {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y/%m/%d %H:%M:%S", &tstruct);

    Json::Value vec(Json::arrayValue);

    est["annot_beats"] = annot_beats;
    est["annot_bounds"] = annot_bounds;
    est["timestamp"] = buf;

    for (int i = 0; i < labels.size(); ++i)
    {
        vec.append(Json::Value(labels[i]));
    }
    est["data"] = vec;
    est["feature"] = feature;
    est["version"] = "1.0";
}

float getFileDur(const char* json_path) {

    string json_str = json_path;
    string jams_path = getJAMSPath(json_str);

    std::filebuf fb;
    fb.open(jams_path.c_str(), ios::in);
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
        return 0;
    }

    float dur = root["metadata"]["duration"].asDouble();

    return dur;
}

vector<int> getAnnotBoundIdxs() {

    std::filebuf fb;
    fb.open("annot_bounds.json", ios::in);
    istream inputs(&fb);

    Json::Value root;   // will contains the root value after parsing.
    Json::Reader reader;
    bool parsingSuccessful = reader.parse( inputs, root );

    vector<int> bounds;
    if ( !parsingSuccessful )
    {
        // report to the user the failure and their locations in the document.
        std::cout  << "Failed to parse configuration\n"
                   << reader.getFormattedErrorMessages();
        return bounds;
    }

    for (int i = 0; i < root["bounds"].size(); i++) {
        bounds.push_back(root["bounds"][i].asInt());
    }
    //Json::Value dataRoot = root["sections"][0]["data"];
    //bounds.push_back(dataRoot[0]["start"]["value"].asDouble());
    //for (int i = 0; i < dataRoot.size(); i++) {
        //bounds.push_back(dataRoot[i]["end"]["value"].asDouble());
    //}

    //for (auto c : bounds)
        //cout << c << " ";
    //cout << endl;

    return bounds;
}

int getFileFrames(const char* json_path) {

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
        return 0;
    }

    return root["est_beatsync"]["hpcp"].size() - 1;
}

void insertDataJSON(Json::Value &root, Json::Value &data, bool bounds, 
        const char* feature, bool annot_beats, bool annot_bounds) {
    
    char data_type[80];
    if (bounds) {
        sprintf(data_type, "boundaries");
    }
    else {
        sprintf(data_type, "labels");
    }

    if (root[data_type]["levy"] == Json::nullValue) {
        Json::Value vec(Json::arrayValue);
        vec.append(data);
        root[data_type]["levy"] = vec;
    }
    else {
        // Find possible value to overwrite
        bool found = false;
        for (int i = 0; i < root[data_type]["levy"].size(); ++i)
        {
            if (strcmp(root[data_type]["levy"][i]["feature"].asCString(), feature) == 0
                && root[data_type]["levy"][i]["annot_beats"].asBool() == annot_beats) {

                if (!root[data_type]["levy"][i].isMember("annot_bounds") ||
                    (root[data_type]["levy"][i].isMember("annot_bounds") &&
                    root[data_type]["levy"][i]["annot_bounds"].asBool() == annot_bounds)) {
                        found = true;
                        root[data_type]["levy"][i] = data;
                        break;
                }
            }
        }
        if (!found) {
            root[data_type]["levy"].append(data);
        }
    }
}

void writeJSON(const char* est_file, vector<double> times, vector<int> labels, bool annot_beats,
               bool annot_bounds, const char* feature) {

    cout << "Writing to: " << est_file << endl;

    std::filebuf fb;
    fb.open(est_file, ios::in);
    istream inputs(&fb);

    Json::Value root;   // will contain the root value after parsing.
    Json::Reader reader;
    bool parsingSuccessful = reader.parse( inputs, root );

    if ( !parsingSuccessful )
    {
        // report to the user the failure and their locations in the document.
        std::cout  << "Failed to parse configuration\n"
                   << reader.getFormattedErrorMessages();
        return;
    }

    Json::Value times_json;
    createEstimation(times_json, times, annot_beats, feature);
    insertDataJSON(root, times_json, true, feature, annot_beats, annot_bounds);

    // Write JSON
    ofstream myfile;
    myfile.open(est_file);
    myfile << root;
    myfile.close();

    parsingSuccessful = reader.parse( inputs, root );
    Json::Value labels_json;
    createEstimationInt(labels_json, labels, annot_beats, annot_bounds, feature);
    insertDataJSON(root, labels_json, false, feature, annot_beats, annot_bounds);

    // cout << root << endl;

    // Write JSON
    myfile.open(est_file);
    myfile << root;
    myfile.close();
}

void writeResults(Segmentation segmentation, bool annot_beats, bool annot_bounds,
                  const char* feats_path, const char *feature) {

    vector<double> beats;
    string feat_path = feats_path;
    float dur = getFileDur(feats_path);

    // If annotated beats, read the beats from the JAMS file
    if (annot_beats) {
        // Get jams file
        string jams_path = getJAMSPath(feat_path);

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

    vector<int> labels;
    labels.resize(segmentation.segments.size());
    for (int i = 0; i < segmentation.segments.size(); ++i)
    {
        labels[i] = segmentation.segments[i].type;
    }

    // Add last segment if needed
    if (!annot_bounds) {
        cout << "last: " << times[segmentation.segments.size()-1] << " " << dur - 0.5 << endl;
        if (times[segmentation.segments.size()-1] < dur - 0.5) {
            times.push_back(dur); 
            if (times.size() - 1 != labels.size())
                labels.push_back(labels[labels.size()-1]);
        }
    }
    //cout << "Times and labels: " << times.size() << " " << labels.size() << endl;

    //for (auto c : times)
        //std::cout << c << ' ';
    //cout << endl;
    //cout << times.size() << endl;
    //for (auto c : labels)
        //std::cout << c << ' ';
    //cout << endl;
    //cout << labels.size() << endl;

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
    writeJSON(est_path.c_str(), times, labels, annot_beats, annot_bounds, feature);
    
}
