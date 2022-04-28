# Visual Analysis application for automatic segmentation algorithms

The web-based visual analysis application allows interactive visual assessment of performances and prediction of automatic segmentation methods.

![visualization overview](https://github.com/CarolineMagg/VA_brain_tumor/blob/main/ui_overview.png)

## Tasks
T1 Performance comparison with performance heatmaps <br>
T2 Relationship to features, i.e., correlaction of performance or model clusters with dataset- and image-derived features <br>
T3 Segmentation masks, i.e., model predictions and GT labels

## Algorithms
The work is connected with [Domain Adaptation for Brain VS segmentation](https://github.com/CarolineMagg/DA_brain). 

## Dataset
The dataset used for this work is publicly available in The Cancer Imaging Archive (TCIA):

Shapey, J., Kujawa, A., Dorent, R., Wang, G., Bisdas, S., Dimitriadis, A., Grishchuck, D., Paddick, I., Kitchen, N., Bradford, R., Saeed, S., Ourselin, S., & Vercauteren, T. (2021). Segmentation of Vestibular Schwannoma from Magnetic Resonance Imaging: An Open Annotated Dataset and Baseline Algorithm [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.9YTJ-5Q73

### How to extract and prepare dataset
Follow instructions from [DA for brain VS segmentation](https://github.com/CarolineMagg/DA_brain/blob/main/README.md).
Note: perform all the steps in Section Dataset.

### Relevant Data split

The split training/test split was 80:20 and the training data is split again into 80:20. The relevant portion of the split is the test set which is used for the evaluation and visualization:

| Dataset    | # samples | numbers (excl.)          |
| ---------- |:---------:| ------------------------:|
| test       | 48        | 200 - 250 (208,219,227)  |

### Prepare results & data

For each patient id, an evaluation.json file is prepared by running `PrepareTestSet`.

### Collect results

The class `TestSet` is used to load the relevant information for the evaluation of different networks (DataContainer for patient data folders, evaluation files per patient folder, evaluation_all.json file if available). In order to reduce loading time at the start of the visualization application, `TestSet` is also able to pre-generate a collection of processed error score values and load them from the file evaluation_all.json. 

## Requirements

The minimal requirements are:
* Python 3.6
* Dash 2.0.0
* Plotly 5.3.1
* Scikit-learn 0.25.4
