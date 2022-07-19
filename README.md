# PPG-BP Analysis
<h3 align="center"> “Can’t Take the Pressure?”: Examining the Challenges of Blood Pressure Estimation via
Pulse Wave Analysis </h3>

<h4 align="center"> Suril Mehta, Nipun Kwatra, Mohit Jain, and Daniel McDuff </h4>

This directory consists of code for analysis of PPG signal for the task of Blood Pressure Estimation via Pulse Wave Analysis

The aim of this README is to describe in detail the setup and image processing pipeline for ease of understanding and usage by a beginner user. The readme describes in detail the analysis pipeline, example code snippets, details on the functions and parameters.

The aim of this README is to provide the code for the tools published in the paper which includes multi-valued mappings and Mutual Information estimation. 


## Dependencies

```
* Pytorch GPU/CPU (torch, torchvision and other dependencies for Pytorch)
```
Please use the environment.yaml file provided in the repo to make an enviroment and install dependencies.

### External Libraries

* [ite-python](https://bitbucket.org/szzoli/ite-in-python/)

## Multi-valued mapping pipeline

The code is present in the multi_valued_mappings folder. The subsequent folders contains the code for each task i.e. Blood Pressure (BP) and Heart Rate (HR). The code for RWAT remains the same as HR. 

### bp
The inputs for this task are:

* `normalized_ppg_path` : Path to the numpy array where each entry consists of all the windows of a particular record. The size of the window needs to be constant. We use a size of 250 which corresponds to 2s of MIMIC data. Each window should be mean-std normalized for the calculation of cross-correlation. Shape of numpy array = (patients, corresponding-windows of each patient, 250)
* `ppg_path` : Similar to normalized_ppg_path but the windows are not mean-std normalized. This is used for the calculation of euclidean distance.  Shape of numpy array = (patients, corresponding-windows of each patient, 250)
* `sbp_path` : Path to the numpy array consisting of SBP values. Shape of numpy array = (patient, corresponding-windows of each patient, 1)
* `abp_path`: Path to the numpy array consisting of the ABP waveforms. Shape of numpy array = (patients, corresponding-windows of each patient, 250)
* `dst_path`: Destination path to store the logs for the script.
* `cross_corr_threshold` : Cross-correlation threshold for filtering. It is kept at 0.9 by default.
* `euclid_threshold` : Euclidean-threshold for filtering.
* `sbp_threshold`: Systolic blood pressure threshold for finding multi-valued mappings. It is kept at 8 by default.
* `max_workers`: The code uses multi-processing to speed up the computation. Ideally it should be kept as num_cpu_cores*2.

#### Running Script

`python inter_patient.py --normalized_ppg_path <norm_ppg_path> --ppg_path <ppg_path> --sbp_path <sbp> --abp_path <abp> --dst_path <logs> --cross_corr_threshold <0.9> --euclid_threshold <1> --sbp_threshold <8> --max_workers <12>`

The code for intra-patient takes the same inputs. 

### hr
The inputs for this task are:

* `normalized_ppg_path` : Path to the numpy array where each entry consists of all the windows of a particular record. The size of the window needs to be constant. We use a size of 250 which corresponds to 2s of MIMIC data. Each window should be mean-std normalized for the calculation of cross-correlation. Shape of numpy array = (patients, corresponding-windows of each patient, 250)
* `ppg_path` : Similar to normalized_ppg_path but the windows are not mean-std normalized. This is used for the calculation of euclidean distance.  Shape of numpy array = (patients, corresponding-windows of each patient, 250)
* `hr_path` : Path to the numpy array consisting of HR values. Shape of numpy array = (patient, corresponding-windows of each patient, 1)
* `dst_path`: Destination path to store the logs for the script.
* `cross_corr_threshold` : Cross-correlation threshold for filtering. It is kept at 0.9 by default.
* `euclid_threshold` : Euclidean-threshold for filtering.
* `hr_threshold`: Systolic blood pressure threshold for finding multi-valued mappings. It is kept at 8 by default.
* `max_workers`: The code uses multi-processing to speed up the computation. Ideally it should be kept as num_cpu_cores*2.

#### Running Script

`python inter_patient.py --normalized_ppg_path <norm_ppg_path> --ppg_path <ppg_path> --hr_path <sbp> --dst_path <logs> --cross_corr_threshold <0.9> --euclid_threshold <1> --hr_threshold <8> --max_workers <12>`

The code for intra-patient takes the same inputs. 

### Interpreting results 

* `inter_patient.py` : The script filters all the multi-valued mappings for each window in a particular record (src_record). The script will dump folders for src_record it is run on. Due to the computation complexity we run it on 100 folders but can be run on the whole dataset.
    * `src_record`: Consists of numpy arrays with which there was a multi-valued mapping was detected (dst_record). These numpy array will consist of the corresponding index of window from src_record and index from dst_record with cross-correlation and euclidean-distance value.

* `intra_patient.py` : The script will create two folders at the dst path *raw* and *val*
    * `raw` : For each record in the ppg_path, a numpy array will be created which will consist of 3 values i.e. total number of windows in the record, total number of multi-valued mappings/collisions in the record, number of indices which had atleast one collision. 
    * `val` : For each record in the ppg_path, a numpy array will be created which will consist of all multi-valued mappings with the 
    src_index, collision_index, cross-correlation value, euclidean-distance value. The array will be empty if there are no multi-valued mappings. 

## Autoencoder
The code is for estimation of mutual information using continous ppg_waveforms.

* `normalized_ppg_path`: Path to the numpy array where each entry consists of all the windows of a particular record. The size of the window needs to be constant. We use a size of 250 which corresponds to 2s of MIMIC data. Each window should be mean-std normalized for the calculation of cross-correlation. Shape of numpy array = (patients, corresponding-windows of each patient, 250)
* `label_path` :  Path to the numpy array consisting of HR / SBP/ RWAT  values. Shape of numpy array = (patient, corresponding-windows of each patient, 1)
* `ite_path` : Path to the information theoretic library. 
* `emb_dim` : Dimensionality of the bottleneck layer that is used for mutual information estimation. 
* `neighbors` : Parameter for mutual information estimation. Higher the value lesser the variance.
* `device` : Device for training either CPU or GPU. If GPU add *cuda:gpu_id*. Default is set to `cuda:0`

#### Running Script
`python main.py --ppg_path <normalized_ppg_path> --label_path <label_values_path> --ite_path <path to information theoretic estimators, present in dependencies> --emb_dim <emb_dim> --neighbours <neighbors> --device <cpu/gpu>`