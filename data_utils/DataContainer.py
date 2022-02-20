########################################################################################################################
# DataContainer to load nifti files in patient data folder
########################################################################################################################
import copy
import json
import logging

import numpy as np
import cv2
import os.path
import nibabel as nib

__author__ = "c.magg"


class DataContainer:
    """
    DataContainer is a container for a nifti folders with different information like:
    * T1
    * T2
    * segmentation of VS/cochlea
    Methods for converting mask (array) into contour (list of coordinates) and vice versa are available.
    """

    def __init__(self, dir_path, alpha=0, beta=1):
        """
        Create a new DataContainer object.
        :param dir_path: path to nifti directory with t1, t2, vs and cochlea segmentation file
        """
        self._path_dir = dir_path
        files = os.listdir(dir_path)
        self._path_t1 = os.path.join(dir_path, f"vs_gk_t1_refT1_processed_{alpha}_{beta}.nii")
        self._path_t2 = os.path.join(dir_path, f"vs_gk_t2_refT1_processed_{alpha}_{beta}.nii")
        self._path_vs = os.path.join(dir_path, "vs_gk_struc1_refT1_processed.nii")
        self._path_vs_margin1 = os.path.join(dir_path, "vs_gk_struc1_refT1_contour1.nii")
        self._path_vs_margin2 = os.path.join(dir_path, "vs_gk_struc1_refT1_contour2.nii")
        self._data_t1 = None
        self._data_t2 = None
        self._data_vs = None
        self._data_margin1 = None
        self._data_margin2 = None
        self.load()

    def __len__(self):
        return self._data_t1.shape[2]

    @property
    def data(self):
        """
        Data dictionary with modality as key and data arrays as values.
        """
        return {'t1': self.t1_scan,
                't2': self.t2_scan,
                'vs': self.vs_segm,
                'margin': self.vs_margin1}

    @property
    def shape(self):
        return self._data_t1.shape

    def load(self):
        """
        (Re)Load the data from nifti paths.
        """
        self._data_t1 = nib.load(self._path_t1)
        self._data_t2 = nib.load(self._path_t2)
        self._data_vs = nib.load(self._path_vs)

    def uncache(self):
        """
        Uncache the nifti container.
        """
        self._data_t1.uncache()
        self._data_t2.uncache()
        self._data_vs.uncache()

    def affine_matrix(self):
        return self._data_vs.affine

    @property
    def t1_scan(self):
        return np.asarray(self._data_t1.dataobj, dtype=np.float32)

    @property
    def t2_scan(self):
        return np.asarray(self._data_t2.dataobj, dtype=np.float32)

    @property
    def vs_segm(self):
        return np.asarray(self._data_vs.dataobj, dtype=np.int16)

    @property
    def vs_margin1(self):
        return np.asarray(self._data_margin1.dataobj, dtype=np.int16)

    @property
    def vs_margin2(self):
        return np.asarray(self._data_margin2.dataobj, dtype=np.int16)

    @property
    def vs_class(self):
        return [1 if np.sum(self.vs_segm[:, :, idx]) != 0 else 0 for idx in range(0, self.vs_segm.shape[2])]

    def t1_scan_slice(self, index=None):
        return np.asarray(self._data_t1.dataobj[..., index], dtype=np.float32)

    def t2_scan_slice(self, index=None):
        return np.asarray(self._data_t2.dataobj[..., index], dtype=np.float32)

    def vs_segm_slice(self, index=None):
        return np.asarray(self._data_vs.dataobj[..., index], dtype=np.int16)

    def vs_margin_slice(self, index=None, thickness=1):
        if thickness == 1:
            return np.asarray(self._data_margin1.dataobj[..., index], dtype=np.int16)
        elif thickness == 2:
            return np.asarray(self._data_margin2.dataobj[..., index], dtype=np.int16)
        else:
            raise ValueError("DataContainer: thickness can only be 1 or 2.")

    def vs_class_slice(self, index=None):
        return self.vs_class[index]

    @staticmethod
    def process_mask_to_contour(segm):
        """
        Create (list of) contours from segmentation mask.
        :param segm: list or array (3D or 2D)
        """
        contours = []
        if type(segm) == list or len(segm.shape) == 3:
            for i in range(len(segm)):
                contour, _ = cv2.findContours(segm[i].astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                contour = [c.tolist() for c in contour]
                contours.append(contour)
        elif len(segm.shape) == 2:
            contours, _ = cv2.findContours(segm.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return contours

    @staticmethod
    def process_contour_to_mask(contour, shape=(256, 256), thickness=1):
        """
        Create (list of) masks from (list of) contours
        :param contour: (list of) contours
        :param shape: shape of array (default: 256x256 - HxW)
        :param thickness: contour thickness (default: 1)
        """
        mask = []
        if type(contour) == list:
            if len(contour) > 0 and type(contour[0]) == list:
                for i in range(len(contour)):
                    m = cv2.drawContours(np.zeros((256, 256)),
                                         [np.array(s).astype(np.int64) for s in contour[i]], -1, 1, 1)
                    mask.append(m)
                mask = np.moveaxis(np.asarray(mask), 0, -1)
            else:
                m = cv2.drawContours(np.zeros((256, 256)),
                                     [np.array(s).astype(np.int64) for s in contour], -1, 1, 1)
                mask.append(m)
                mask = np.moveaxis(np.asarray(mask), 0, -1)
            mask = mask.astype(np.int16)
        else:
            raise TypeError("DataContainer: contour object needs to be of type list.")
        return mask
