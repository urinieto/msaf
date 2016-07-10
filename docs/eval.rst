.. _eval:

Evaluations
===========

MSAF includes the standard evaluation metrics used in `MIREX <http://www.music-ir.org/mirex/wiki/2016:Structural_Segmentation>`_. Here we describe each of these metrics, classified based on the subtask they aim to assess.

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
