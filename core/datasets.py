import os
import sys
from typing import Dict, Optional, List

import torch
from sklearn.base import BaseEstimator
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
)

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CORE_DIR)

sys.path.append(ROOT_DIR)

from core.data_model import *


AVAILABLE_SCALERS = {
    "standard": StandardScaler,
    "min_max": MinMaxScaler,
    "robust": RobustScaler,
    "max_abs": MaxAbsScaler,
}


class MILDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        patient_datasets: PatientDataSet,
        scaler: Optional[BaseEstimator] = None,
        device="cuda",
        gnomad_ac: dict = {},
        clinvar_variants: dict = {},
    ):
        self.patient_datasets = patient_datasets
        self.scaler = scaler
        self.device = device

        self.gnomad_ac = gnomad_ac
        self.clinvar_variants = clinvar_variants

    def __len__(self):
        return len(self.patient_datasets)

    def __getitem__(self, idx):
        patient_data: PatientDataSet = self.patient_datasets[idx]

        # TODO remove
        ac = np.zeros((len(patient_data.data[0].snv_data.variants), 1))
        n_benign = np.zeros((len(patient_data.data[0].snv_data.variants), 1))
        for idx, variant in enumerate(patient_data.data[0].snv_data.variants):
            ac[idx] = self.gnomad_ac.get(variant.cpra, 0)
            n_benign[idx] = self.clinvar_variants.get(variant.cpra, 0)

        x = np.concatenate([patient_data.snv_x, ac, n_benign], axis=1).astype(np.float32)

        if self.scaler is not None:
            x = self.scaler.transform(x)

        x, instance_label, bag_label = map(
            lambda x: torch.from_numpy(x)
            .to(
                self.device,
            )
            .float(),
            [
                x,
                patient_data.snv_instance_y.astype(np.float32),
                np.array([patient_data.bag_labels], dtype=np.float32),
            ],
        )
        return x, instance_label, bag_label


class ExSCNVDataset(torch.utils.data.Dataset):
    """External SNV + CNV Dataset: Extra feature가 있는 데이터셋"""

    def __init__(
        self,
        patient_datasets: PatientDataSet,
        base_features: List[str],
        additional_features: List[str],
        scalers: dict = dict(),
        device="cuda",
    ):
        self.patient_datasets = patient_datasets
        self.base_features = base_features
        self.additional_features = additional_features
        self.scalers = scalers
        self.device = device
        self._set_feature_indices()

    def _set_feature_indices(self) -> None:
        """Additional features을 concat하기위해서 인덱스를 구함"""
        for patient_data in self.patient_datasets:
            self.feature_indices = np.array(
                [
                    patient_data.snv_data.header.index(feat)
                    for feat in self.base_features + self.additional_features
                ]
            )

            break

        return

    def __len__(self):
        return len(self.patient_datasets)

    def __getitem__(self, idx):
        patient_data: PatientData = self.patient_datasets[idx]

        instance_label = np.concatenate(
            [
                patient_data.snv_data.y.astype(np.float32),
                patient_data.cnv_data.y.astype(np.float32),
            ]
        )

        snv_x = patient_data.snv_data.x[:, self.feature_indices].astype(np.float32)
        cnv_x = patient_data.cnv_data.x.astype(np.float32)

        if self.scalers:
            snv_x = self.scalers["snv"].transform(snv_x)
            cnv_x = self.scalers["cnv"].transform(cnv_x)

        snv_x, cnv_x, instance_label, bag_label = map(
            lambda x: torch.from_numpy(x).to(self.device).float(),
            [
                snv_x,
                cnv_x,
                instance_label,
                np.array([patient_data.bag_label]),
            ],
        )
        return (snv_x, cnv_x), instance_label, bag_label


class MultimodalMILDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        patient_datasets: PatientDataSet,
        scaler: Optional[Dict[str, BaseEstimator]] = {"snv": None, "cnv": None},
        device="cuda",
    ):
        self.patient_datasets = patient_datasets
        self.scaler = scaler
        self.device = device

        self.snv_scaler = self.scaler.get("snv")
        self.cnv_scaler = self.scaler.get("cnv")

    def __len__(self):
        return len(self.patient_datasets)

    def __getitem__(self, idx):
        patient_data: PatientDataSet = self.patient_datasets[idx]
        instance_label = np.concatenate(
            [
                patient_data.snv_instance_y.astype(np.float32),
                patient_data.cnv_instance_y.astype(np.float32),
            ]
        )

        snv_x = patient_data.snv_x.astype(np.float32)
        cnv_x = patient_data.cnv_x.astype(np.float32)

        if self.snv_scaler is not None:
            snv_x = self.snv_scaler.transform(snv_x)
        if self.cnv_scaler is not None:
            cnv_x = self.cnv_scaler.transform(cnv_x)

        snv_x, cnv_x, instance_label, bag_label = map(
            lambda x: torch.from_numpy(x).to(self.device).float(),
            [
                snv_x,
                cnv_x,
                instance_label,
                patient_data.bag_labels,
            ],
        )
        return (snv_x, cnv_x), instance_label, bag_label
