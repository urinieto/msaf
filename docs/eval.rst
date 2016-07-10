.. _eval:

Evaluations
===========

MSAF includes the standard evaluation metrics used in `MIREX <http://www.music-ir.org/mirex/wiki/2016:Structural_Segmentation>`_. Here we describe how to evaluate the algorithms' results and discuss each of these metrics, classified based on the subtask they aim to assess.

How To Evaluate Results
-----------------------

The module `eval.py <https://github.com/urinieto/msaf/blob/master/msaf/eval.py>`_ contains the following ``process`` function that can be called once the desired algorithms have been run on a single file or dataset:

.. automodule:: msaf.eval

The return value of this function is a dictionary (or a list of dictionaries, in case of *collection mode*) containing all of the available metrics for the evaluated subtask(s).
The keys to this dictionary, with a description of each metric are found below.

Boundary Metrics
----------------

=================  ==============
Boundary Metric    Description
=================  ==============
D                  Information Gain  
DevE2R             Median Deviation from Estimation to Reference 
DevR2E             Median Deviation from Reference to Estimation 
DevtE2R            Median Deviation from Estimation to Reference without first and last boundaries (trimmed)
DevtR2E            Median Deviation from Reference to Estimation without first and last boundaries (trimmed)
HitRate\_0.5F      Hit Rate F-measure using 0.5 seconds window 
HitRate\_0.5P      Hit Rate Precision using 0.5 seconds window 
HitRate\_0.5R      Hit Rate Recall using 0.5 seconds window 
HitRate\_3F        Hit Rate F-measure using 3 seconds window 
HitRate\_3P        Hit Rate Precision using 3 seconds window 
HitRate\_3R        Hit Rate Recall using 3 seconds window 
HitRate\_t0.5F     Hit Rate F-measure using 0.5 seconds window without first and last boundaries (trimmed)
HitRate\_t0.5P     Hit Rate Precision using 0.5 seconds window without first and last boundaries (trimmed)
HitRate\_t0.5R     Hit Rate Recall using 0.5 seconds window without first and last boundaries (trimmed)
HitRate\_t3F       Hit Rate F-measure using 3 seconds window without first and last boundaries (trimmed)
HitRate\_t3P       Hit Rate Precision using 3 seconds window without first and last boundaries (trimmed)
HitRate\_t3R       Hit Rate Recall using 3 seconds window without first and last boundaries (trimmed)
t_measure10        T-Measures F-measure at 10 seconds window
t_precision10      T-Measures Precision at 10 seconds window
t_recall10         T-Measures Recall at 10 seconds window
t_measure15        T-Measures F-measure at 15 seconds window
t_precision15      T-Measures Precision at 15 seconds window
t_recall15         T-Measures Recall at 15 seconds window
=================  ==============

Label Metrics
-------------

=================  ==============
Label Metric       Description
=================  ==============
PWF                Pairwise Frame Clustering F-measure 
PWP                Pairwise Frame Clustering Precision 
PWR                Pairwise Frame Clustering Recall 
Sf                 Normalized Entropy Scores F-measure 
So                 Normalized Entropy Scores Precision 
Su                 Normalized Entropy Scores Recall 
=================  ==============
