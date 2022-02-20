########################################################################################################################
# TestSet containing information about test dataset
########################################################################################################################
import json
import logging
import os
import time

import pandas as pd
import numpy as np
from natsort import natsorted
from sklearn.metrics import confusion_matrix

from data_utils.DataContainer import DataContainer

__author__ = "c.magg"


class TestSet:
    """
    TestSet is a container for the test set information:
    * T1 and T2 images
    * ground truth for VS segmentation
    * predictions for VS segmentation
    * pandas dataframe with information
    """

    MODELS_SIMPLE1 = ["XNet_T2_relu", "XNet_T2_leaky", "XNet_T2_selu"]
    MODELS_SIMPLE2 = ["XNet_T1_relu", "XNet_T1_leaky", "XNet_T1_selu"]
    MODELS_SIMPLE = [*MODELS_SIMPLE1, *MODELS_SIMPLE2]
    MODELS_CG = ["CG_XNet_T1_relu", "CG_XNet_T2_relu"]
    MODELS_DA = ["SegmS2T_GAN1_relu", "SegmS2T_GAN2_relu", "SegmS2T_GAN5_relu",
                 "CG_SegmS2T_GAN1_relu", "CG_SegmS2T_GAN2_relu", "CG_SegmS2T_GAN5_relu"]
    MODELS_GAN = ["GAN_1+XNet_T1_relu", "GAN_2+XNet_T1_relu", "GAN_5+XNet_T1_relu",
                  "GAN_1+CG_XNet_T1_relu", "GAN_2+CG_XNet_T1_relu", "GAN_5+CG_XNet_T1_relu"]
    MODELS = [*MODELS_SIMPLE, *MODELS_CG, *MODELS_GAN, *MODELS_DA]

    exclude_ids = [228, 229, 242, 243, 249]  # exluding IDs since no orthonormal direction cosines

    def __init__(self, path="/tf/workdir/data/VS_segm/test_processed/",
                 dsize=(256, 256), load=True, data_load=False, evaluation_load=False, radiomics_load=False,
                 full_mask=True):
        self._path = path
        self._full_mask = full_mask
        self._folders = natsorted(os.listdir(path))
        self._dsize = dsize
        self._data_container_0_1 = []
        self._data_container_1_1 = []
        self._df_evaluation = []
        self._df_radiomics_3d = []
        self._df_radiomics_3d_margin = []
        self._df_radiomics_2d = []
        self._patient_ids = []
        self._patient_ids_radiomics = []
        self._df_total = pd.DataFrame()
        self._df_signature_3d = pd.DataFrame()
        self._df_signature_3d_margin = pd.DataFrame()
        self._df_signature_2d = pd.DataFrame()
        self._df_volume = pd.DataFrame()

        if load:
            self.load_data(data_load, evaluation_load, radiomics_load)

    @staticmethod
    def lookup_data_call():
        """
        Generate the mapping between data naming and data loading method in DataContainer.
        :return lookup dictionary with data name as key and method name as value
        """
        return {'t1': 't1_scan_slice',
                't2': 't2_scan_slice',
                'vs': 'segm_vs_slice',
                'margin': 'vs_margin_slice'}

    @property
    def all_models(self):
        return self.MODELS

    @property
    def mask_mode(self):
        return True if self._full_mask else False

    @mask_mode.setter
    def mask_mode(self, mask_mode):
        if mask_mode == "full_mask" or mask_mode is True:
            self._full_mask = True
        elif mask_mode == "margin_mask" or mask_mode is False:
            self._full_mask = False
        else:
            raise ValueError("TestSet: unvalid mask mode.")

    def load_data(self, data=False, evaluation=True, radiomics=True):
        """
        Load the following data:
        * nifti files
        * prediction information
        * evaluation information (if available)
        * all radiomics information (if available)
        * volumetric information (if available)

        :param data:
        :param evaluation:
        :param radiomics:
        :return:
        """
        logging.info("Load data.")
        patient_ids = []
        patient_ids_radiomics = []
        if data or evaluation:
            for f in self._folders:
                folder = os.path.join(self._path, f)
                patient_id = f.split("/")[-1].split("_")[-1]
                # load NIFTI data
                if data:
                    self._data_container_0_1.append(DataContainer(folder, alpha=0, beta=1))
                    self._data_container_1_1.append(DataContainer(folder, alpha=-1, beta=1))
                # load prediction information
                if evaluation:
                    self._df_evaluation.append(pd.read_json(os.path.join(folder, "evaluation.json")))
                if radiomics:
                    if os.path.isfile(os.path.join(folder, "radiomics.json")):
                        with open(os.path.join(folder, "radiomics.json")) as json_file:
                            self._df_radiomics_3d.append(json.load(json_file))
                        patient_ids_radiomics.append(patient_id)
                    if os.path.isfile(os.path.join(folder, "radiomics_2d.json")):
                        with open(os.path.join(folder, "radiomics_2d.json")) as json_file:
                            self._df_radiomics_2d.append(json.load(json_file))
                    if os.path.isfile(os.path.join(folder, "radiomics_gt_contour1.json")):
                        with open(os.path.join(folder, "radiomics_gt_contour1.json")) as json_file:
                            self._df_radiomics_3d_margin.append(json.load(json_file))
                patient_ids.append(patient_id)
            self._patient_ids = patient_ids
            self._patient_ids_radiomics = patient_ids_radiomics

        file_path = "/tf/workdir/tu_vienna/VA_brain_tumor/data_utils/evaluation_all.json"
        if os.path.isfile(file_path):
            logging.info("Load all evaluation information.")
            self._df_total = pd.read_json(file_path)

        file_path = "/tf/workdir/tu_vienna/VA_brain_tumor/data_utils/radiomics_all.json"
        if os.path.isfile(file_path):
            logging.info("Load all radiomics features for full mask.")
            self._df_signature_3d = pd.read_json(file_path)

        file_path = "/tf/workdir/tu_vienna/VA_brain_tumor/data_utils/radiomics_all_margin.json"
        if os.path.isfile(file_path):
            logging.info("Load all radiomics features for margin mask.")
            self._df_signature_3d_margin = pd.read_json(file_path)

        file_path = "/tf/workdir/tu_vienna/VA_brain_tumor/data_utils/volumentric.json"
        if os.path.isfile(file_path):
            logging.info("Load volumetric information.")
            self._df_volume = pd.read_json(file_path)

    def get_patient_data(self, idx, alpha=0, beta=1):
        if alpha == 0 and beta == 1:
            return self._data_container_0_1[idx]
        elif alpha == -1 and beta == 1:
            return self._data_container_1_1[idx]

    @staticmethod
    def calculate_accuracy(conf_mat):
        if type(conf_mat) == dict:
            return (conf_mat["tp"] + conf_mat["tn"]) / (
                    conf_mat["tp"] + conf_mat["tn"] + conf_mat["fp"] + conf_mat["fn"])
        else:
            return (conf_mat[3] + conf_mat[0]) / (conf_mat[3] + conf_mat[0] + conf_mat[1] + conf_mat[2])

    @staticmethod
    def calculate_tpr(conf_mat):
        if type(conf_mat) == dict:
            return (conf_mat["tp"]) / (conf_mat["tp"] + conf_mat["fn"])
        else:
            return (conf_mat[3]) / (conf_mat[3] + conf_mat[2])

    @staticmethod
    def calculate_tnr(conf_mat):
        if type(conf_mat) == dict:
            return (conf_mat["tn"]) / (conf_mat["tn"] + conf_mat["fp"]) if conf_mat["tn"] != 0 else 1.0
        else:
            return (conf_mat[0]) / (conf_mat[0] + conf_mat[1]) if conf_mat[0] != 0 else 1.0

    @property
    def list_patient_ids(self):
        patient_ids = self._patient_ids
        if len(patient_ids) == 0:
            for f in self._folders:
                patient_ids.append(f.split("/")[-1].split("_")[-1])
        return patient_ids

    @property
    def list_patient_ids_radiomics(self):
        patient_ids = self._patient_ids_radiomics
        if len(patient_ids) == 0:
            for f in self._folders:
                if os.path.isfile(os.path.join(self._path, f, "radiomics.json")):
                    patient_ids.append(f.split("/")[-1].split("_")[-1])
        return patient_ids

    def _generate_signature_3d(self, df_radiomics_3d):
        # postprocess folder-structure to have a list of values
        df_radiomics = {"id": self._patient_ids_radiomics}
        feature_classes = list(df_radiomics_3d[0].keys())
        for cl in feature_classes:
            cl_dict = {}
            for key in range(len(self._patient_ids_radiomics)):
                cl_dict[key] = df_radiomics_3d[key][cl]
            tmp = {}
            for idx, d in cl_dict.items():
                for f, vals in d.items():
                    if f in tmp.keys():
                        tmp[f] = tmp[f] + [vals]
                    else:
                        tmp[f] = [vals]
            df_radiomics[cl] = tmp
        # generate signature
        df_sign = pd.DataFrame(columns=["id"])
        df_sign["id"] = df_radiomics["id"]
        for fc in feature_classes:
            for key, vals in df_radiomics[fc].items():
                vals = [float(v) for v in vals]
                if key == "Skewness":
                    df_sign[f"{fc}-{key}"] = [1 if a <= 0 else 2 for a in vals]
                elif key == "Kurtosis":
                    df_sign[f"{fc}-{key}"] = [1 if a <= 3 else 2 for a in vals]
                elif key == "Elongation":
                    df_sign[f"{fc}-{key}"] = [1 if a <= np.mean(vals) else 2 for a in vals]
                elif key == "Flatness":
                    df_sign[f"{fc}-{key}"] = [1 if a <= 0.5 else 2 for a in vals]
                elif key == "Sphericity":
                    df_sign[f"{fc}-{key}"] = [1 if a <= np.mean(vals) else 2 for a in vals]
                else:
                    df_sign[f"{fc}-{key}"] = np.digitize(vals, bins=np.linspace(np.min(vals),
                                                                                np.nextafter(np.max(vals), np.inf),
                                                                                4))
        return df_sign

    @property
    def df_signature_3d(self):
        if self._full_mask:
            if len(self._df_signature_3d) == 0:
                logging.info("Generate signature df for full mask.")
                self._df_signature_3d = self._generate_signature_3d(self._df_radiomics_3d)
            return self._df_signature_3d
        else:
            if len(self._df_signature_3d_margin) == 0:
                logging.info("Generate signature df for margin mask.")
                self._df_signature_3d_margin = self._generate_signature_3d(self._df_radiomics_3d_margin)
            return self._df_signature_3d_margin

    @property
    def dict_signature_2d(self):
        return self._df_radiomics_2d

    @property
    def df_total(self):
        if len(self._df_total) == 0:
            logging.info("Generate total df.")
            # all slices
            df_dice_all = pd.DataFrame(columns=["id", *self.MODELS])
            df_assd_all = pd.DataFrame(columns=["id", *self.MODELS])
            df_conf_mat_all = pd.DataFrame(columns=["id", *self.MODELS])
            df_acc_all = pd.DataFrame(columns=["id", *self.MODELS])
            df_tpr_all = pd.DataFrame(columns=["id", *self.MODELS])
            df_tnr_all = pd.DataFrame(columns=["id", *self.MODELS])
            df_dice_all["id"] = [*self._patient_ids, "DSC"]
            df_assd_all["id"] = [*self._patient_ids, "ASSD"]
            df_conf_mat_all["id"] = [*self._patient_ids, "conf_mat"]
            df_acc_all["id"] = [*self._patient_ids, "ACC"]
            df_tpr_all["id"] = [*self._patient_ids, "TPR"]
            df_tnr_all["id"] = [*self._patient_ids, "TNR"]
            # only tumor slices
            df_dice_only_tumor = pd.DataFrame(columns=["id", *self.MODELS])
            df_assd_only_tumor = pd.DataFrame(columns=["id", *self.MODELS])
            df_conf_mat_only_tumor = pd.DataFrame(columns=["id", *self.MODELS])
            df_acc_only_tumor = pd.DataFrame(columns=["id", *self.MODELS])
            df_tpr_only_tumor = pd.DataFrame(columns=["id", *self.MODELS])
            df_tnr_only_tumor = pd.DataFrame(columns=["id", *self.MODELS])
            df_dice_only_tumor["id"] = [*self._patient_ids, "DSC"]
            df_assd_only_tumor["id"] = [*self._patient_ids, "ASSD"]
            df_conf_mat_only_tumor["id"] = [*self._patient_ids, "conf_mat"]
            df_acc_only_tumor["id"] = [*self._patient_ids, "ACC"]
            df_tpr_only_tumor["id"] = [*self._patient_ids, "TPR"]
            df_tnr_only_tumor["id"] = [*self._patient_ids, "TNR"]

            for name in self.MODELS:
                dice_list = []
                dice_list_tumor = []
                assd_list = []
                assd_list_tumor = []
                tp_list = []
                tn_list = []
                fn_list = []
                fp_list = []
                tp_list_tumor = []
                tn_list_tumor = []
                fn_list_tumor = []
                fp_list_tumor = []
                cols_number = df_dice_all.columns.get_loc(name)
                for idx, df in enumerate(self._df_evaluation):
                    # all slices
                    df_dice_all.iloc[idx, cols_number] = np.mean(df[f"VS_segm_dice-{name}"].values)
                    df_assd_all.iloc[idx, cols_number] = np.mean(df[f"VS_segm_assd-{name}"].values)
                    dice_list += list(df[f"VS_segm_dice-{name}"].values)
                    assd_list += list(df[f"VS_segm_assd-{name}"].values)
                    tn, fp, fn, tp = confusion_matrix(df["VS_class_gt"].values,
                                                      df[f"VS_class_pred-{name}"].values,
                                                      labels=[0, 1]).ravel()
                    tp_list.append(tp)
                    tn_list.append(tn)
                    fn_list.append(fn)
                    fp_list.append(fp)
                    conf_mat_all = {"tp": tp, "tn": tn, "fn": fn, "fp": fp}
                    df_conf_mat_all.iloc[idx, cols_number] = str(conf_mat_all)
                    df_acc_all.iloc[idx, cols_number] = self.calculate_accuracy(conf_mat_all)
                    df_tpr_all.iloc[idx, cols_number] = self.calculate_tpr(conf_mat_all)
                    df_tnr_all.iloc[idx, cols_number] = self.calculate_tnr(conf_mat_all)
                    # only tumor slices
                    df_dice_only_tumor.iloc[idx, cols_number] = np.mean(
                        df[df["VS_class_gt"] == 1][f"VS_segm_dice-{name}"].values)
                    df_assd_only_tumor.iloc[idx, cols_number] = np.mean(
                        df[df["VS_class_gt"] == 1][f"VS_segm_assd-{name}"].values)
                    dice_list_tumor += list(df[df["VS_class_gt"] == 1][f"VS_segm_dice-{name}"].values)
                    assd_list_tumor += list(df[df["VS_class_gt"] == 1][f"VS_segm_assd-{name}"].values)
                    tn, fp, fn, tp = confusion_matrix(df[df["VS_class_gt"] == 1]["VS_class_gt"].values,
                                                      df[df["VS_class_gt"] == 1][f"VS_class_pred-{name}"].values,
                                                      labels=[0, 1]).ravel()
                    tp_list_tumor.append(tp)
                    tn_list_tumor.append(tn)
                    fn_list_tumor.append(fn)
                    fp_list_tumor.append(fp)
                    conf_mat = {"tp": tp, "tn": tn, "fn": fn, "fp": fp}
                    df_conf_mat_only_tumor.iloc[idx, cols_number] = str(conf_mat)
                    df_acc_only_tumor.iloc[idx, cols_number] = self.calculate_accuracy(conf_mat)
                    df_tpr_only_tumor.iloc[idx, cols_number] = self.calculate_tpr(conf_mat)
                    df_tnr_only_tumor.iloc[idx, cols_number] = self.calculate_tnr(conf_mat)
                df_dice_all.iloc[-1, cols_number] = np.mean(dice_list)
                df_assd_all.iloc[-1, cols_number] = np.mean(assd_list)
                conf_mat_all = {"tp": np.sum(tp_list), "tn": np.sum(tn_list),
                                "fn": np.sum(fn_list), "fp": np.sum(fp_list)}
                df_acc_all.iloc[-1, cols_number] = self.calculate_accuracy(conf_mat_all)
                df_tpr_all.iloc[-1, cols_number] = self.calculate_tpr(conf_mat_all)
                df_tnr_all.iloc[-1, cols_number] = self.calculate_tnr(conf_mat_all)
                df_conf_mat_all.iloc[-1, cols_number] = str(conf_mat_all)
                df_dice_only_tumor.iloc[-1, cols_number] = np.mean(dice_list_tumor)
                df_assd_only_tumor.iloc[-1, cols_number] = np.mean(assd_list_tumor)
                conf_mat = {"tp": np.sum(tp_list_tumor), "tn": np.sum(tn_list_tumor),
                            "fn": np.sum(fn_list_tumor), "fp": np.sum(fp_list_tumor)}
                df_conf_mat_only_tumor.iloc[-1, cols_number] = str(conf_mat)
                df_acc_only_tumor.iloc[-1, cols_number] = self.calculate_accuracy(conf_mat)
                df_tpr_only_tumor.iloc[-1, cols_number] = self.calculate_tpr(conf_mat)
                df_tnr_only_tumor.iloc[-1, cols_number] = self.calculate_tnr(conf_mat)
            collect_df = pd.DataFrame()
            collect_df = collect_df.append({"dice_all": df_dice_all,
                                            "dice_only_tumor": df_dice_only_tumor,
                                            "assd_all": df_assd_all,
                                            "assd_only_tumor": df_assd_only_tumor,
                                            "conf_mat_all": df_conf_mat_all,
                                            "conf_mat_only_tumor": df_conf_mat_only_tumor,
                                            "acc_all": df_acc_all,
                                            "acc_only_tumor": df_acc_only_tumor,
                                            "tpr_all": df_tpr_all,
                                            "tpr_only_tumor": df_tpr_only_tumor,
                                            "tnr_all": df_tnr_all,
                                            "tnr_only_tumor": df_tnr_only_tumor}, ignore_index=True)
            self._df_total = collect_df
        return self._df_total

    @property
    def df_volume(self):
        df_vol = self._df_volume
        if len(self._df_volume) == 0:
            df_vol = pd.DataFrame()
            for idx, patient_id in enumerate(testset.list_patient_ids):
                df = self._df_evaluation[idx]
                df_vol = df_vol.append({"id": patient_id,
                                        "slice_number": len(df),
                                        "tumor_slice_number": len(df[df["VS_class_gt"] == 1])}, ignore_index=True)
            self._df_volume = df_vol
        return df_vol

    @property
    def df_features(self):
        df_feature = pd.DataFrame()

        return df_feature

    def __len__(self):
        return len(self._data_container_0_1)


if __name__ == "__main__":
    testset = TestSet("/tf/workdir/data/VS_segm/test_processed/", load=True,
                      data_load=True, evaluation_load=True, radiomics_load=True)
    start = time.time()
    df_total = testset.df_total
    # df_total.to_json("/tf/workdir/tu_vienna/VA_brain_tumor/data_utils/evaluation_all.json")
    print("df total", time.time() - start)
    start = time.time()
    df_signature = testset.df_signature_3d
    # df_signature.to_json("/tf/workdir/tu_vienna/VA_brain_tumor/data_utils/radiomics_all.json")
    print("df signature", time.time() - start)
    start = time.time()
    df_volume = testset.df_volume
    # df_volume.to_json("/tf/workdir/tu_vienna/VA_brain_tumor/data_utils/volumentric.json")
    print("df volume", time.time() - start)
    start = time.time()
    testset.mask_mode = "margin_mask"
    df_signature_mask = testset.df_signature_3d
    # df_signature_mask.to_json("/tf/workdir/tu_vienna/VA_brain_tumor/data_utils/radiomics_all_margin.json")
    print("df signature margin", time.time() - start)
