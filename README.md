# vcog_hps_ad

This repository contains packages, scripts, and notebooks for the following article:
Angela Tam, Christian Dansereau, Yasser Iturria-Medina, Sebastian Urchs, Pierre Orban, Hanad Sharmarke, John Breitner, Pierre Bellec, Alzheimer's Disease Neuroimaging Initiative, A highly predictive signature of cognition and brain atrophy for progression to Alzheimer's dementia, GigaScience, Volume 8, Issue 5, May 2019, giz055, [https://doi.org/10.1093/gigascience/giz055](https://doi.org/10.1093/gigascience/giz055)

Click the following link to reproduce the analysis with simulated data on binder: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SIMEXP/vcog_hps_ad/master?filepath=%2Fvcog_hpc_prediction_simulated_data.ipynb)  
Here is a brief description of each item in the repository:
* **adas13_mixed_effects.ipynb** -  a Jupyter notebook that gives the linear mixed effects models for cognitive trajectories of different groups
* **adni_bl_vbm_pipeline_20171201.m** - an Octave script that runs a segmentation pipeline from SPM12 inside a NIAK container
* **adni_csv_merging.ipynb** - a Jupyter notebook that merges ADNI spreadsheets together
* **adni_filter_mci_csv.ipynb** - a Jupyter notebook that filters eligible MCI subjects
* **cog_hpc_prediction.ipynb** - a Jupyter notebook containing analyses that give a  highly predictive signature (HPS) of Alzheimer's disease dementia using cognitive features that were derived from real data
* **Proteus** - a Python package by [Christian Dansereau](https://github.com/cdansereau). Proteus was built on [scikit-learn](http://scikit-learn.org/stable/#) and it offers machine learning tools to make highly confident predictions
* **vbm_hpc_prediction.ipynb** - a Jupyter notebook containing analyses that give a highly predictive signature (HPS) of Alzheimer's disease dementia using structural features that were derived from real data
* **vbm_subtypes_glm.ipynb** - a Jupyter notebook that provides univariate tests between vbm subtypes and diagnosis
* **vbm_subtypes_pipeline.m** - an Octave script to build subtypes of grey matter atrophy and extract weights from structural T1 images
* **vcog_hpc_prediction.ipynb** - a Jupyter notebook containing analyses that give a highly predictive signature (HPS) of Alzheimer's disease dementia from cognitive and structural brain features that were derived from real data
* **vcog_hpc_prediction_simulated_data.ipynb** - a Jupyter notebook containing analyses that give a highly predictive signature (HPS) of Alzheimer's disease dementia from cognitive and structural features using *simulated data*
* **simulation_script.py** - a Python script that generates simulated data from raw data 
* **simulated_data.csv** - a comma separated value file that contains simulated data 
* **spm_container** - an Octave package containing wrappers for [SPM12](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/) functions for segmentation and DARTEL 

This repository has also been archived on Zenodo: [![DOI](https://zenodo.org/badge/129415986.svg)](https://zenodo.org/badge/latestdoi/129415986)
 

 
 

 
 
  








