/*
Little program to run the segmenter using the MSAF features

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"
*/


#include <Python.h>
#include <numpy/arrayobject.h>
#include "ClusterMeltSegmenter.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <time.h>

using namespace std;

static PyObject* segment(PyObject *self, PyObject *args);

static PyMethodDef CCMethods[] = {
    {"segment",  segment, METH_VARARGS, 
        "Segments an audio file using the Constrained Clustering method."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


PyMODINIT_FUNC initcc_segmenter(void)
{
    PyObject *m = Py_InitModule3("cc_segmenter", CCMethods,
            "Module to segment audio files using the Constrained Clustering method");

    if (m == NULL)
        return;

    /* Load `numpy` functionality. */
    import_array();
}

/*
 * Assigns the data in array "data" to the 2D vector v, of sizes N x M.
 */
void assignDataTo2dVector(double *data, vector<vector<double> > &v, int N, int M) {
    for (int i = 0; i < N; i++) {
        v[i].assign(&data[i*M], &data[i*M + M]);
    }
}

static PyObject* segment(PyObject *self, PyObject *args) {

    PyObject *features_obj;
    PyObject *in_bounds_obj;
    npy_intp *shape_features;
    int is_harmonic;
    int nHMMStates;
    int nclusters;
    int neighbourhoodLimit;
    int sample_rate;

    if (!PyArg_ParseTuple(args, "iiiiiOO", &is_harmonic, &nHMMStates, &nclusters,
                &neighbourhoodLimit, &sample_rate, &features_obj, &in_bounds_obj)) {
        return NULL;
    }

    // Get numpy arrays
    PyObject *features_array = PyArray_FROM_OTF(features_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *in_bounds_array = PyArray_FROM_OTF(in_bounds_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    if (features_array == NULL || in_bounds_array == NULL) {
        Py_XDECREF(features_array);
        Py_XDECREF(in_bounds_array);
        return NULL;
    }

    // Get numpy array dimensions
    shape_features = PyArray_DIMS(features_array);
    int N = (int)PyArray_DIM(features_array, 0);
    int M = (int)shape_features[1];
    int in_bounds_N = (int)PyArray_DIM(in_bounds_array, 0);

//    printf("Features Size N: %d, M: %d\n", N, M);
//    printf("sample rate: %d\n", sample_rate);
//    printf("nHMMStates: %d\n", nHMMStates);
//    printf("nclusters: %d\n", nclusters);
//    printf("neighbourhoodLimit: %d\n", neighbourhoodLimit);
//    printf("is_harmonic: %d\n", is_harmonic);

    // Numpy Arrays to vectors
    double *features_data = (double*)PyArray_DATA(features_array);
    vector<vector<double> > f(N, vector<double>(M));
    assignDataTo2dVector(features_data, f, N, M);

    // Input boundaries might be empty
    double *in_bounds_data = (double*)PyArray_DATA(in_bounds_array);
    vector<int> in_bounds(in_bounds_N);
    in_bounds.assign(in_bounds_data, in_bounds_data + in_bounds_N);

    //printf("In Bounds Len: %d\n", in_bounds_N);
    //for (int i = 0; i < in_bounds_N; i++) {
        //cout << in_bounds[i] << " ";
    //}
    //cout << endl;

    Segmentation segmentation;
    ClusterMeltSegmenterParams params;

    if (is_harmonic) {
        params.featureType = FEATURE_TYPE_CHROMA;
    }
    else {
        params.featureType = FEATURE_TYPE_MFCC;
    }

    // Set parameters
    params.nHMMStates = nHMMStates;
    params.nclusters = nclusters;
    params.neighbourhoodLimit = neighbourhoodLimit;

    ClusterMeltSegmenter *segmenter = new ClusterMeltSegmenter(params);

    // Segment until we have a potentially good result
    int tries = 0;
    do {
        // Initialize segmenter
        segmenter->initialise(sample_rate);

        // Set features
        segmenter->setFeatures(f);

        // Set previously computed Boundaries Indeces if needed
        if (in_bounds_N > 0) {
            segmenter->setAnnotBounds(in_bounds);
        }

        // Segment
        segmenter->segment();
        segmentation = segmenter->getSegmentation();
        tries++;

    } while(segmentation.segments.size() < 2 && f.size() >= 90 && tries < 10);


    //cout << segmentation.segments.size() << endl;
//    cout << "estimated labels: ";
//    for (auto b : segmentation.segments) {
//        cout << b.type << " ";
//    }
//    cout << endl;

    // Put segmentation times in a Python List
    int times_len = segmentation.segments.size() + 1;
    PyObject *times = PyList_New(times_len);
    if (!times)
        return NULL;
    PyObject *num = PyInt_FromLong(segmentation.segments[0].start);
    PyList_SET_ITEM(times, 0, num);
    for (int i = 0; i < times_len - 1; i++) {
        PyObject *num = PyInt_FromLong(segmentation.segments[i].end);
        if (!num) {
            Py_DECREF(times);
            return NULL;
        }
        PyList_SET_ITEM(times, i + 1, num);
    }

    // Put Labels in a Python list
    int labels_len = segmentation.segments.size();
    PyObject *labels = PyList_New(labels_len);
    if (!labels)
        return NULL;
    for (int i = 0; i < labels_len; i++) {
        PyObject *num = PyInt_FromLong(segmentation.segments[i].type);
        if (!num) {
            Py_DECREF(labels);
            return NULL;
        }
        PyList_SET_ITEM(labels, i, num);
    }

    // Cleanup
    Py_XDECREF(features_array);
    Py_XDECREF(in_bounds_array);
    delete segmenter;

    // Return Python tuple (est_times, est_labels)
    return Py_BuildValue("[O,O]", times, labels);
}
