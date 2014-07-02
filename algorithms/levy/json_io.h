/** Some functions to read and write JSON files.
 *
 * author: Oriol Nieto, 2014
 * **/

#ifndef __JSON_IO__H
#define __JSON_IO__H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <time.h>
#include <json/json.h>
#include "ClusterMeltSegmenter.h"

using namespace std;

/** Transform JSON Value to a 2D Vector (vector<vector<double> >) **/
vector<vector<double> > JSONValueTo2DVector(Json::Value &data);

/** Transform JSON Value to a 1D Vector (vector<double>) **/
vector<double> JSONValueToVector(Json::Value &data);
vector<double> JAMSValueToVector(Json::Value &data);
vector<vector<double> > readJSON(const char* json_path, bool annot_beats,
                        const char* feature);
vector<double> readBeatsJAMS(const char *jams_path);

/** Get properties from JSON/JAMS **/
string getJAMSPath(string feat_path);
float getFileDur(const char* json_path);
int getFileFrames(const char* json_path);
vector<int> getAnnotBoundIdxs();

void createEstimation(Json::Value &est, vector<double> &times, 
        bool annot_beats, const char* feature);
void createEstimationInt(Json::Value &est, vector<int> &labels, 
        bool annot_beats, bool annot_bounds, const char* feature);

/** Write to JSON **/
void insertDataJSON(Json::Value &root, Json::Value &data, bool bounds, 
        const char* feature, bool annot_beats, bool annot_bounds);
void writeJSON(const char* est_file, vector<double> times, vector<int> labels, bool annot_beats, 
               bool annot_bounds, const char* feature);
void writeResults(Segmentation segmentation, bool annot_beats, bool annot_bounds,
                  const char* feats_path, const char *feature);

#endif
