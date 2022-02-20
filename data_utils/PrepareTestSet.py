########################################################################################################################
# Script to prepare the data for evaluation & visualization
#
# * generate processed data and store as NIFTI to "test_processed" folder
# * generate predictions for selected list of models with model inference and store:
#       * contours (list of coordinates)
#       * error metrics (e.g., dice, assd)
#   per subject into json files
# * generate radiomics features for 3D and 2D data with filled GT as ROI (radiomics.json and radiomics_2d.json)
# * generate GT contour masks with thickness 1 and 2 and store them as Nifti files to "test_processed" folder
# * generate radiomics features for 3D data with contour GT as ROI(radiomics_gt_contour.json)
########################################################################################################################

import json
import os
import shutil
import time
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from medpy import metric
import nibabel as nib
from natsort import natsorted
from radiomics import featureextractor
import radiomics
import SimpleITK as sitk

from data_utils.DataContainer import DataContainer
from model_zoo.losses.dice import DiceLoss, DiceCoefficient
from model_zoo.utils import check_gpu

__author__ = "c.magg"


class DataTransformer:
    """Process data and store as NIFTI"""

    def __init__(self, dir_path, dsize=(256, 256), alpha=0, beta=1):
        self._path_dir = dir_path
        files = os.listdir(dir_path)
        self._path_t1 = os.path.join(dir_path, "vs_gk_t1_refT1.nii")
        self._path_t2 = os.path.join(dir_path, "vs_gk_t2_refT1.nii")
        self._path_vs = os.path.join(dir_path, [f for f in files if "vs_gk_struc1_" in f and "processed" not in f][0])
        self._path_data_info = os.path.join(dir_path, "vs_gk_statistics.json")
        self._statistics = None
        self._dsize = dsize
        self._alpha = alpha
        self._beta = beta

        # pre-load data
        self._data_t1 = nib.load(self._path_t1)
        self._data_t2 = nib.load(self._path_t2)
        self._data_vs = nib.load(self._path_vs)
        with open(self._path_data_info) as json_file:
            self._statistics = json.load(json_file)

    def transform_and_store_data(self):
        """
        Transform and store data
        * load and process data
        * generate new filename
        * store as nifti
        """
        t1_scan = self._load_image(np.asarray(self._data_t1.dataobj, dtype=np.float32), "t1")
        t2_scan = self._load_image(np.asarray(self._data_t2.dataobj, dtype=np.float32), "t2")
        vs_segm = self._load_segm(np.asarray(self._data_vs.dataobj, dtype=np.int16))
        fn_t1 = self._data_t1.get_filename().replace(".nii", f"_processed_{self._alpha}_{self._beta}.nii").replace(
            "test/", "test_processed/")
        fn_t2 = self._data_t2.get_filename().replace(".nii", f"_processed_{self._alpha}_{self._beta}.nii").replace(
            "test/", "test_processed/")
        fn_vs = os.path.join(os.path.dirname(self._data_vs.get_filename()), "vs_gk_struc1_refT1_processed.nii").replace(
            "test/", "test_processed/")
        nib.save(nib.Nifti1Image(t1_scan, self._data_t1.affine), fn_t1)
        nib.save(nib.Nifti1Image(t2_scan, self._data_t2.affine), fn_t2)
        nib.save(nib.Nifti1Image(vs_segm, self._data_vs.affine), fn_vs)

    def _load_segm(self, array):
        """
        Load segmentation
        * resize
        """
        return cv2.resize(array, dsize=self._dsize, interpolation=cv2.INTER_NEAREST)

    def _load_image(self, array, data_type):
        """
        Load image
        process data volume:
        * clip extrem values
        * z score normalization
        * range [0,1]
        process each slice
        * resize
        * normalize [alpha, beta]
        """
        sample = np.clip(array, float(self._statistics[data_type]["1st_percentile"]),
                         float(self._statistics[data_type]["99th_percentile"]))
        sample = (sample - float(self._statistics[data_type]["mean"])) / float(self._statistics[data_type]["std"])
        sample = (sample - float(self._statistics[data_type]["min"])) / (
                float(self._statistics[data_type]["max"]) - float(self._statistics[data_type]["min"]))

        sample = np.moveaxis(sample, 2, 0)
        data = np.zeros((len(sample), 256, 256))
        for k in range(len(sample)):
            data[k] = cv2.resize(sample[k], dsize=self._dsize, interpolation=cv2.INTER_CUBIC)
            if np.max(data[k]) - np.min(data[k]) == 0:
                data[k] = data[k]
            else:
                data[k] = ((data[k] - np.min(data[k])) / (np.max(data[k]) - np.min(data[k]))) * (
                        self._beta - self._alpha) + self._alpha
        return np.moveaxis(data, 0, 2)


MODELS = {"XNet_T2_relu": {"path": "/tf/workdir/DA_vis/model_zoo/XNet_t2_relu_segm_13318/",
                           "custom_objects": {'DiceLoss': DiceLoss, 'DiceCoefficient': DiceCoefficient}},
          "XNet_T2_leaky": {"path": "/tf/workdir/DA_vis/model_zoo/XNet_t2_leaky_relu_segm_13318/",
                            "custom_objects": {'DiceLoss': DiceLoss, 'DiceCoefficient': DiceCoefficient}},
          "XNet_T2_selu": {"path": "/tf/workdir/DA_vis/model_zoo/XNet_t2_selu_segm_13318/",
                           "custom_objects": {'DiceLoss': DiceLoss, 'DiceCoefficient': DiceCoefficient}},
          "XNet_T1_relu": {"path": "/tf/workdir/DA_vis/model_zoo/XNet_t1_relu_segm_13318/",
                           "custom_objects": {'DiceLoss': DiceLoss, 'DiceCoefficient': DiceCoefficient}},
          "XNet_T1_leaky": {"path": "/tf/workdir/DA_vis/model_zoo/XNet_t1_leaky_relu_segm_13318/",
                            "custom_objects": {'DiceLoss': DiceLoss, 'DiceCoefficient': DiceCoefficient}},
          "XNet_T1_selu": {"path": "/tf/workdir/DA_vis/model_zoo/XNet_t1_selu_segm_13318/",
                           "custom_objects": {'DiceLoss': DiceLoss, 'DiceCoefficient': DiceCoefficient}},
          "CG_XNet_T2_relu": {"path": "/tf/workdir/DA_vis/model_zoo/cg_XNet_t2_relu_True_segm_13319/",
                              "custom_objects": {'DiceLoss': DiceLoss, 'DiceCoefficient': DiceCoefficient}},
          "CG_XNet_T1_relu": {"path": "/tf/workdir/DA_vis/model_zoo/cg_XNet_t1_relu_True_segm_13319/",
                              "custom_objects": {'DiceLoss': DiceLoss, 'DiceCoefficient': DiceCoefficient}},
          "SegmS2T_GAN1_relu": {"path": "/tf/workdir/DA_vis/model_zoo/segmS2T_XNet_relu_150_200_gan_1_13319_v2/",
                                "custom_objects": None},
          "SegmS2T_GAN2_relu": {"path": "/tf/workdir/DA_vis/model_zoo/segmS2T_XNet_relu_150_200_gan_2_13319_v2/",
                                "custom_objects": None},
          "SegmS2T_GAN5_relu": {"path": "/tf/workdir/DA_vis/model_zoo/segmS2T_XNet_relu_150_200_gan_5_13319_v2/",
                                "custom_objects": None},
          "CG_SegmS2T_GAN1_relu": {
              "path": "/tf/workdir/DA_vis/model_zoo/cgsegmS2T_XNet_relu_150_200_gan_1_13319_v2/CGSegmS2T",
              "custom_objects": None},
          "CG_SegmS2T_GAN2_relu": {
              "path": "/tf/workdir/DA_vis/model_zoo/cgsegmS2T_XNet_relu_150_200_gan_2_13319_v2/CGSegmS2T",
              "custom_objects": None},
          "CG_SegmS2T_GAN5_relu": {
              "path": "/tf/workdir/DA_vis/model_zoo/cgsegmS2T_XNet_relu_150_200_gan_5_13319_v2/CGSegmS2T",
              "custom_objects": None},
          "GAN_1": {"path": "/tf/workdir/DA_vis/model_zoo/gan_1_100_50_13785/G_T2S",
                    "custom_objects": None},
          "GAN_2": {"path": "/tf/workdir/DA_vis/model_zoo/gan_2_100_50_13786/G_T2S",
                    "custom_objects": None},
          "GAN_5": {"path": "/tf/workdir/DA_vis/model_zoo/gan_5_100_50_13786/G_T2S",
                    "custom_objects": None},
          }

MODELS_CG = ["CG_XNet_T1_relu", "CG_XNet_T2_relu",
             "CG_SegmS2T_GAN1_relu", "CG_SegmS2T_GAN2_relu", "CG_SegmS2T_GAN5_relu"]
MODELS_GAN = ["GAN_1", "GAN_2", "GAN_5"]
MODELS_SIMPLE = ["XNet_T2_relu", "XNet_T2_leaky", "XNet_T2_selu",
                 "SegmS2T_GAN1_relu", "SegmS2T_GAN2_relu", "SegmS2T_GAN5_relu",
                 "XNet_T1_relu", "XNet_T1_leaky", "XNet_T1_selu"]


def simple_predict(model, img, gt):
    """Generate simple prediction"""
    res = model.predict(np.expand_dims(img, axis=0))[0, :, :, 0]
    _, res_proc = cv2.threshold(res, 0.5, 1, cv2.THRESH_BINARY)
    d = 0
    sd = 362
    if np.sum(gt) == 0 and np.sum(res_proc) == 0:
        d = 1
        sd = 0
    elif np.sum(gt) != 0 and np.sum(res_proc) != 0:
        d = metric.binary.dc(res_proc, gt)
        sd = metric.binary.assd(res_proc, gt)
    return res_proc, d, sd


def cg_predict(model, img, gt):
    """Generate classification-guided predictions"""
    res1, res2 = model.predict(np.expand_dims(img, axis=0))
    _, res_proc = cv2.threshold(res1[0, :, :, 0], 0.5, 1, cv2.THRESH_BINARY)
    res2_proc = res2[0][0]
    d = 0
    sd = 362
    if np.sum(gt) == 0 and np.sum(res_proc) == 0:
        d = 1
        sd = 0
    elif np.sum(gt) != 0 and np.sum(res_proc) != 0:
        d = metric.binary.dc(res_proc, gt)
        sd = metric.binary.assd(res_proc, gt)
    return res_proc, res2_proc, d, sd


def gan_predict(model1, model2, img, gt):
    """Generate CycleGAN-based prediction - 2 steps in inference"""
    res_intermediate = model1.predict(np.expand_dims(img, axis=0))
    res = model2.predict((res_intermediate + 1) / 2)
    _, res_proc = cv2.threshold(res[0, :, :, 0], 0.5, 1, cv2.THRESH_BINARY)
    d = 0
    sd = 362
    if np.sum(gt) == 0 and np.sum(res_proc) == 0:
        d = 1
        sd = 0
    elif np.sum(gt) != 0 and np.sum(res_proc) != 0:
        d = metric.binary.dc(res_proc, gt)
        sd = metric.binary.assd(res_proc, gt)
    return res_proc, res_intermediate, d, sd


def gan_cg_predict(model1, model2, img, gt):
    """Generate CycleGAN-based prediction - 2 steps in inference"""
    res_intermediate = model1.predict(np.expand_dims(img, axis=0))
    res1, res2 = model2.predict((res_intermediate + 1) / 2)
    _, res_proc = cv2.threshold(res1[0, :, :, 0], 0.5, 1, cv2.THRESH_BINARY)
    res2_proc = res2[0][0]
    d = 0
    sd = 362
    if np.sum(gt) == 0 and np.sum(res_proc) == 0:
        d = 1
        sd = 0
    elif np.sum(gt) != 0 and np.sum(res_proc) != 0:
        d = metric.binary.dc(res_proc, gt)
        sd = metric.binary.assd(res_proc, gt)
    return res_proc, res2_proc, res_intermediate, d, sd


if __name__ == "__main__":

    check_gpu()

    data_path = "/tf/workdir/data/VS_segm"

    DO_DATA_TRANSFORM = False  # perform pre-processing (clip, resize, ...) and store Nifti files
    DO_PRED_GENERATION = False  # generate predictions, extract error metrics and store prediction contour in json
    DO_RADIOMICS_3D = False  # get 3D radiomics from original T2 and filled GT masks and store json file
    DO_RADIOMICS_2D = False  # get 2D radiomics from original T2 and filled GT masks and store json file
    DO_GT_CONTOUR_GENERATION = True  # generate GT contours with thickness 1 and 2 and store Nifti files
    DO_GT_CONTOUR_RADIOMICS_3D = True  # get 3D radiomics from original T2 and contour GT masks and store json file

    if DO_DATA_TRANSFORM:
        path = f"{data_path}/test/"
        folders = natsorted([os.path.join(path, f) for f in os.listdir(path)])
        print("Transform data.")
        start = time.time()
        for f in folders:
            if not os.path.exists(f.replace("test/", "test_processed/")):
                os.mkdir(f.replace("test/", "test_processed/"))
            # load data
            container = DataTransformer(f, alpha=0, beta=1)
            container2 = DataTransformer(f, alpha=-1, beta=1)
            # transforms and store data
            container.transform_and_store_data()
            container2.transform_and_store_data()
        print(f"Finish transform data in {time.time() - start} sec.")

    if DO_PRED_GENERATION:
        path = f"{data_path}/test_processed/"
        folders = natsorted([os.path.join(path, f) for f in os.listdir(path)])
        print("Generate predictions.")
        start = time.time()
        for f in folders:
            # load data
            container = DataContainer(f, alpha=0, beta=1)
            t2_scan = np.moveaxis(container.t2_scan, 2, 0)
            vs_segm = np.moveaxis(container.vs_segm, 2, 0)
            vs_class = container.vs_class
            container2 = DataContainer(f, alpha=-1, beta=1)
            t2_scan2 = np.moveaxis(container2.t2_scan, 2, 0)
            t1_scan2 = np.moveaxis(container2.t1_scan, 2, 0)
            # store general information
            df = pd.DataFrame(columns=["slice", "VS_segm_gt", "VS_class_gt"])
            df["slice"] = list(range(len(container)))
            df["VS_segm_gt"] = container.process_mask_to_contour(vs_segm)
            df["VS_class_gt"] = vs_class

            for name in MODELS_SIMPLE:
                model = tf.keras.models.load_model(MODELS[name]["path"],
                                                   custom_objects=MODELS[name]["custom_objects"])
                pred_segm = []
                dice = []
                surface_distance = []
                for idx, img in enumerate(t2_scan):
                    gt = vs_segm[idx]
                    res_proc, dsc, assd = simple_predict(model, img, gt)
                    pred_segm.append(res_proc)
                    surface_distance.append(assd)
                    dice.append(dsc)
                df[f"VS_segm_pred-{name}"] = container.process_mask_to_contour(pred_segm)
                df[f"VS_segm_dice-{name}"] = dice
                df[f"VS_segm_assd-{name}"] = surface_distance
                df[f"VS_class_pred-{name}"] = [1 if np.sum(p) != 0 else 0 for p in pred_segm]

            for name in MODELS_CG:
                model = tf.keras.models.load_model(MODELS[name]["path"],
                                                   custom_objects=MODELS[name]["custom_objects"])
                pred_segm = []
                pred_class_prob = []
                dice = []
                surface_distance = []
                for idx, img in enumerate(t2_scan):
                    gt = vs_segm[idx]
                    res_proc, res2, dsc, assd = cg_predict(model, img, gt)
                    pred_segm.append(res_proc)
                    pred_class_prob.append(res2)
                    surface_distance.append(assd)
                    dice.append(dsc)
                df[f"VS_segm_pred-{name}"] = container.process_mask_to_contour(pred_segm)
                df[f"VS_segm_dice-{name}"] = dice
                df[f"VS_segm_assd-{name}"] = surface_distance
                df[f"VS_class_pred-{name}"] = [1 if np.sum(p) != 0 else 0 for p in pred_segm]
                df[f"VS_class_pred_prob-{name}"] = pred_class_prob

            for name in MODELS_GAN:
                model1 = tf.keras.models.load_model(MODELS[name]["path"],
                                                    custom_objects=MODELS[name]["custom_objects"])
                model2 = tf.keras.models.load_model(MODELS["XNet_T1_relu"]["path"],
                                                    custom_objects=MODELS["XNet_T1_relu"]["custom_objects"])
                pred_segm = []
                dice = []
                surface_distance = []
                mse = []
                for idx, img in enumerate(t2_scan2):
                    gt = vs_segm[idx]
                    res_proc, res_intermediate, dsc, assd = gan_predict(model1, model2, img, gt)
                    mse.append(tf.reduce_mean(tf.keras.losses.mean_squared_error(res_intermediate,
                                                                                 tf.expand_dims(t1_scan2[idx],
                                                                                                -1))).numpy())
                    pred_segm.append(res_proc)
                    surface_distance.append(assd)
                    dice.append(dsc)
                df[f"VS_segm_pred-{name}+XNet_T1_relu"] = container.process_mask_to_contour(pred_segm)
                df[f"VS_segm_dice-{name}+XNet_T1_relu"] = dice
                df[f"VS_segm_mse-{name}+XNet_T1_relu"] = mse
                df[f"VS_segm_assd-{name}+XNet_T1_relu"] = surface_distance
                df[f"VS_class_pred-{name}+XNet_T1_relu"] = [1 if np.sum(p) != 0 else 0 for p in pred_segm]

            for name in MODELS_GAN:
                model1 = tf.keras.models.load_model(MODELS[name]["path"],
                                                    custom_objects=MODELS[name]["custom_objects"])
                model2 = tf.keras.models.load_model(MODELS["CG_XNet_T1_relu"]["path"],
                                                    custom_objects=MODELS["CG_XNet_T1_relu"]["custom_objects"])
                pred_segm = []
                dice = []
                surface_distance = []
                mse = []
                pred_class_prob = []
                for idx, img in enumerate(t2_scan2):
                    gt = vs_segm[idx]
                    res_proc, res2, res_intermediate, dsc, assd = gan_cg_predict(model1, model2, img, gt)
                    mse.append(tf.reduce_mean(tf.keras.losses.mean_squared_error(res_intermediate,
                                                                                 tf.expand_dims(t1_scan2[idx],
                                                                                                -1))).numpy())
                    pred_segm.append(res_proc)
                    surface_distance.append(assd)
                    dice.append(dsc)
                    pred_class_prob.append(res2)
                df[f"VS_segm_pred-{name}+CG_XNet_T1_relu"] = container.process_mask_to_contour(pred_segm)
                df[f"VS_segm_dice-{name}+CG_XNet_T1_relu"] = dice
                df[f"VS_segm_mse-{name}+CG_XNet_T1_relu"] = mse
                df[f"VS_segm_assd-{name}+CG_XNet_T1_relu"] = surface_distance
                df[f"VS_class_pred-{name}+CG_XNet_T1_relu"] = [1 if np.sum(p) != 0 else 0 for p in pred_segm]
                df[f"VS_class_pred_prob-{name}+CG_XNet_T1_relu"] = pred_class_prob

            df.to_json(os.path.join(f, "evaluation.json"))

        print(f"Finish generate predictions in {time.time() - start} sec.")

    if DO_RADIOMICS_3D:
        path = f"{data_path}]/test_processed/"
        folders = natsorted([os.path.join(path, f) for f in os.listdir(path)])
        print("Extract 3D radiomics features.")
        # extract features
        start = time.time()
        settings = {'binWidth': 25, 'resampledPixelSpacing': None, 'interpolator': sitk.sitkBSpline}
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        featureClasses = radiomics.getFeatureClasses()
        featureClasses.pop("shape2D", "None")
        for folder in folders:
            imageName = os.path.join(folder, "vs_gk_t2_refT1_processed_0_1.nii")
            maskName = os.path.join(folder, "vs_gk_struc1_refT1_processed.nii")
            featureVectorProc = {}
            try:
                featureVector = extractor.execute(imageName, maskName)
                tmp = {}
                for idx, featureName in enumerate(featureVector.keys()):
                    if f"original_" in featureName and "-original_" not in featureName:
                        tmp[featureName.split('_')[-1] + "-" + featureName.split('_')[-2]] = featureVector[
                                                                                                 featureName] * 1
                for fc in featureClasses:
                    featureVectorProc[fc] = {k.split("-")[0]: str(v) for k, v in tmp.items() if fc in k}
                with open(os.path.join(folder, "radiomics.json"), 'w') as outfile:
                    json.dump(featureVectorProc, outfile)
            except RuntimeError as e:
                print("error ", folder.split("/")[-1])
        print(time.time() - start)

    if DO_RADIOMICS_2D:
        path = f"{data_path}/test_processed/"
        folders = natsorted([os.path.join(path, f) for f in os.listdir(path)])
        print("Extract 2D radiomics features.")
        # extract features
        start = time.time()
        settings = {'binWidth': 25, 'resampledPixelSpacing': None, 'interpolator': sitk.sitkBSpline}
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        extractor.enableAllFeatures()
        for folder in folders:
            # generate tmp 2D data
            tmp_folder = os.path.join(folder, "tmp")
            if not os.path.isdir(tmp_folder):
                os.mkdir(tmp_folder)
            container = DataContainer(folder)
            for slice_id in range(len(container)):
                img = container.t2_scan_slice(slice_id)
                segm = container.vs_segm_slice(slice_id)
                fn_img = os.path.join(tmp_folder, f"t2_{slice_id}.nii")
                fn_segm = os.path.join(tmp_folder, f"slice_{slice_id}.nii")
                nib.save(nib.Nifti1Image(img, None), fn_img)
                nib.save(nib.Nifti1Image(segm, None), fn_segm)
            # generate 2D features
            files = natsorted([os.path.join(tmp_folder, f) for f in os.listdir(tmp_folder) if "t2" in f])
            featureVectorProcList = {}
            for file in files:
                imageName = file
                maskName = file.replace("t2", "slice")
                try:
                    featureVector = extractor.execute(imageName, maskName)
                    tmp = {}
                    featureVectorProc = {}
                    for idx, featureName in enumerate(featureVector.keys()):
                        if f"original_" in featureName and "-original_" not in featureName:
                            tmp[featureName.split('_')[-1] + "-" + featureName.split('_')[-2]] = featureVector[
                                                                                                     featureName] * 1
                    featureClasses = radiomics.getFeatureClasses()
                    for fc in featureClasses:
                        featureVectorProc[fc] = {k.split("-")[0]: str(v) for k, v in tmp.items() if fc in k}
                    featureVectorProcList[file.split("_")[-1].split(".")[0]] = featureVectorProc
                except ValueError:
                    continue
            with open(os.path.join(folder, "radiomics_2d.json"), 'w') as outfile:
                json.dump(featureVectorProcList, outfile)
            # remove tmp folder again
            shutil.rmtree(tmp_folder)

        print(time.time() - start)

    if DO_GT_CONTOUR_GENERATION:
        path = f"{data_path}/test_processed/"
        folders = natsorted([os.path.join(path, f) for f in os.listdir(path)])
        print("Generate GT contour masks.")
        start = time.time()
        for folder in folders:
            container = DataContainer(folder)
            affine_matrix = container.affine_matrix()
            df = pd.read_json(os.path.join(folder, "evaluation.json"))
            segm_contour = container.process_contour_to_mask(list(df["VS_segm_gt"].values), thickness=1)
            nib.save(nib.Nifti1Image(segm_contour, affine_matrix), f"{folder}/vs_gk_struc1_refT1_contour1")
            segm_contour = container.process_contour_to_mask(list(df["VS_segm_gt"].values), thickness=2)
            nib.save(nib.Nifti1Image(segm_contour, affine_matrix), f"{folder}/vs_gk_struc1_refT1_contour2")

        print(time.time() - start)

    if DO_GT_CONTOUR_RADIOMICS_3D:
        path = f"{data_path}/test_processed/"
        folders = natsorted([os.path.join(path, f) for f in os.listdir(path)])
        print("Generate GT contour masks.")
        start = time.time()
        settings = {'binWidth': 25, 'resampledPixelSpacing': None, 'interpolator': sitk.sitkBSpline}
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        featureClasses = radiomics.getFeatureClasses()
        featureClasses.pop("shape2D", "None")
        for folder in folders:
            imageName = os.path.join(folder, "vs_gk_t2_refT1_processed_0_1.nii")
            maskName = os.path.join(folder, "vs_gk_struc1_refT1_contour1.nii")
            featureVectorProc = {}
            try:
                featureVector = extractor.execute(imageName, maskName)
                tmp = {}
                for idx, featureName in enumerate(featureVector.keys()):
                    if f"original_" in featureName and "-original_" not in featureName:
                        tmp[featureName.split('_')[-1] + "-" + featureName.split('_')[-2]] = featureVector[
                                                                                                 featureName] * 1
                for fc in featureClasses:
                    featureVectorProc[fc] = {k.split("-")[0]: str(v) for k, v in tmp.items() if fc in k}
                with open(os.path.join(folder, "radiomics_gt_contour1.json"), 'w') as outfile:
                    json.dump(featureVectorProc, outfile)
            except RuntimeError as e:
                print("error ", folder.split("/")[-1])

            maskName = os.path.join(folder, "vs_gk_struc1_refT1_contour2.nii")
            featureVectorProc = {}
            try:
                featureVector = extractor.execute(imageName, maskName)
                tmp = {}
                for idx, featureName in enumerate(featureVector.keys()):
                    if f"original_" in featureName and "-original_" not in featureName:
                        tmp[featureName.split('_')[-1] + "-" + featureName.split('_')[-2]] = featureVector[
                                                                                                 featureName] * 1
                for fc in featureClasses:
                    featureVectorProc[fc] = {k.split("-")[0]: str(v) for k, v in tmp.items() if fc in k}
                with open(os.path.join(folder, "radiomics_gt_contour2.json"), 'w') as outfile:
                    json.dump(featureVectorProc, outfile)
            except RuntimeError as e:
                print("error ", folder.split("/")[-1])

        print(time.time() - start)


