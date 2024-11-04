import os
import sys
from typing import List, Union

import tqdm
import numpy as np
from mlflow.entities import Run
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split

REV_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.dirname(REV_DIR)
ROOT_DIR = os.path.dirname(RES_DIR)

sys.path.append(ROOT_DIR)
from core.data_model import PatientDataSet, PatientData

TRACKING_URI = ""
EXP_NAME_MIL = "MIL_multimodal"
DATA_PATH = "/data2/heon_dev/repository/3ASC-Confirmed-variant-Resys/dataset_positive_negative.pickle"
DATA_PATH = "/home/heon/dev/ASC3/dataset_positive_negative.pickle"
ORIGINAL_REPO_EXP_DIR = (
    "/data/heon_dev/repository/3ASC-Confirmed-variant-Resys/experiments"
)
RANDOM_STATE = 20230524

BOOSTRAP_RUN_ID = "07397e4fda5f4dcdb6e98438382428a0"
RF_RUN_ID = "4c2634a1990f454f8b70dbd05687c5f1"
GENE_SPLIT_ID = "f1dab109b0354a128bec0a2274001d2f"
RUN_ASC3_W_RANKNET = "29f1dfc8ee2b464cb6ec36627f12e429"
RUN_ASC3_WO_RANKNET = "87cec01ac82a42a6a6fffce3b7543597"
ASC3_V1_RUN_ID = "13b463ffd5ac4d8ebb1f2062cd3cbe19"

FOLD_RESULTS_PICKLE = "mlflow-artifacts:/6/{run_id}/artifacts/fold_result.pickle"

BM_DIR = "/data/heon_dev/repository/3ASC-Confirmed-variant-Resys/notebooks/MIL"
DISEASE_DATA = (
    "/DAS/data/personal/sean_dev/misc/20241017.3asc_revision/result/disease.txt.gz"
)


def open_pickle(path):
    import pickle

    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def get_child_runs(parent_run_id) -> List[Run]:
    client = MlflowClient()

    all_runs = client.search_runs(
        experiment_ids=[6],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
    )

    child_runs = []
    for run in all_runs:
        child_runs.append(run.info.run_id)

    return child_runs


def split_by_gene(patient_dataset, random_state: Union[int, None] = 20230524) -> tuple:
    """
    환자1 = (Gene 1)
    환자2 = (Gene 2)
    환자3 = (Gene 3, 4)
    환자4 = (Gene 4, 5)
    환자5 = (Gene 6)
    환자6 = (Gene 7)
    환자7 = Negative
    환자8 = Negative

    SNV
    G = {G_training, G_test}
    G_training = confirmed variant의 gene을 모두 하나라도 포함된 경우 training dataset내에 포함되어있는경우
    G_test = G - G_training

    Example
    1) G_training = {Gene 1, 2, 4} => 환자 1, 2, 3, 4
    2) G_test = 환자 5, 6
    3) 환자7 -> training, 환자 8->test
    """

    # confirmed gene id 수집
    negative_samples = list()
    cnv_only_samples = list()
    snv_only_samples = list()
    confirmed_gene_ids = set()
    for patient_data in tqdm.tqdm(patient_dataset):
        if not patient_data.bag_label:
            negative_samples.append(patient_data)
            continue

        # Label=True
        if patient_data.snv_data.causal_variant == [("-", "-")]:
            cnv_only_samples.append(patient_data)
            continue

        # Confirmed variant의 GeneId만 수집
        confirmed_variants = {
            cpra for (cpra, disease_id) in patient_data.snv_data.causal_variant
        }
        for snv in patient_data.snv_data.variants:
            if snv.cpra in confirmed_variants:
                confirmed_gene_ids.add(snv.gene_id)

        snv_only_samples.append(patient_data)

    # train-test-split: Positive case에 한함
    train_gene_ids, test_gene_ids = train_test_split(
        list(confirmed_gene_ids), test_size=0.2, random_state=random_state
    )
    train_patient_data = list()
    test_patient_data = list()
    for i, patient_data in tqdm.tqdm(enumerate(snv_only_samples)):
        confirmed_variants = {
            cpra for (cpra, disease_id) in patient_data.snv_data.causal_variant
        }

        is_assigned = False
        for snv in patient_data.snv_data.variants:
            if snv.cpra in confirmed_variants:  # 원인변이
                if snv.gene_id in train_gene_ids:
                    train_patient_data.append(patient_data)
                    is_assigned = True
                    break
                else:
                    test_patient_data.append(patient_data)
                    is_assigned = True
                    break

    # CNV patient split
    train_cnv_samples, test_cnv_samples = train_test_split(cnv_only_samples)
    train_negative_samples, test_negative_samples = train_test_split(negative_samples)

    train_dataset = (
        train_patient_data + list(train_cnv_samples) + list(train_negative_samples)
    )
    test_dataset = (
        test_patient_data + list(test_cnv_samples) + list(test_negative_samples)
    )
    return PatientDataSet(train_dataset), PatientDataSet(test_dataset)


def get_topk_folds(
    fold_y_trues: List[List[np.ndarray]],
    fold_y_probs: List[List[np.ndarray]],
    ks=[1, 2, 3, 4, 5, 10, 15, 20, 100],
):
    from core.metric import topk_recall

    n_folds = len(fold_y_trues)
    performance = np.ones((n_folds, len(ks)))

    for k_idx, k in enumerate(ks):
        for fold_idx, (instance_y_trues_at_fold, instance_y_probs_at_fold) in enumerate(
            zip(
                fold_y_trues,
                fold_y_probs,
            )
        ):
            hits = list()
            for instance_y_true, instance_y_probs in zip(
                instance_y_trues_at_fold, instance_y_probs_at_fold
            ):
                if instance_y_true.sum() == 0:
                    continue

                hit = topk_recall(instance_y_probs, instance_y_true, k=k)
                hits.append(hit)

            performance[fold_idx, k_idx] = sum(hits) / len(hits)

    return performance


def cohen_d(group1, group2):
    # 두 그룹의 평균과 표준 편차 계산
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    # 그룹의 결합 표준 편차 계산
    pooled_std = np.sqrt(
        ((len(group1) - 1) * std1**2 + (len(group2) - 1) * std2**2)
        / (len(group1) + len(group2) - 2)
    )

    # Cohen's d 계산
    d = (mean1 - mean2) / pooled_std
    return d


def get_snv_only_case(patient_dataset):
    test_ids = list()
    for i, patient in enumerate(patient_dataset):
        if patient.bag_label == False:
            continue

        if patient.snv_data.causal_variant == [("-", "-")]:
            continue

        if patient.cnv_data.causal_variant:
            continue

        test_ids.append(patient.sample_id)

    return test_ids


def get_cnv_only_case(patient_dataset):
    test_ids = list()
    for i, patient in enumerate(patient_dataset):
        if patient.bag_label == False:
            continue

        if patient.snv_data.causal_variant != [("-", "-")]:
            continue

        if not patient.cnv_data.causal_variant:
            continue

        test_ids.append(patient.sample_id)

    return test_ids


def get_negative_case(patient_dataset):
    test_ids = list()
    for i, patient in enumerate(patient_dataset):
        if patient.bag_label:
            continue

        test_ids.append(patient.sample_id)

    return test_ids


def get_patient_with_inheritance(
    patient_dataset, disease2inheritance: dict, pattern="autosomal dominant"
):
    test_ids = list()
    for i, patient in enumerate(patient_dataset):
        patient: PatientData = patient
        disease_ids = [
            omim_id.lstrip("OMIM:")
            for (cpra, omim_id) in patient.snv_data.causal_variant
        ]
        inheritances = [
            disease2inheritance.get(disease_id, str()).lower()
            for disease_id in disease_ids
        ]
        is_match = any([inheritance == pattern for inheritance in inheritances])
        if is_match:
            test_ids.append(patient.sample_id)

    return patient_dataset[test_ids]


def benchmark_tool_topk(df, casual_variants: list, k: int, score_col=str()) -> bool:
    _df = df.copy()
    if not score_col:
        _df = _df.sort_values("score", ascending=False).reset_index(drop=True)

    if len(_df.loc[_df["cpra"].isin(casual_variants)]) == 0:
        return False

    rank = min(_df.loc[_df["cpra"].isin(casual_variants)].index.tolist())
    if rank > k:
        return False

    return True


def benchmark_exomsier(df, casual_variants: list, k: int, score_col) -> bool:
    _df = df.copy()
    if not score_col:
        _df = _df.sort_values("score", ascending=False).reset_index(drop=True)

    if len(_df.loc[_df["cpra"].isin(casual_variants)]) == 0:
        return False

    rank = min(_df.loc[_df["cpra"].isin(casual_variants)].index.tolist())
    if rank > k:
        return False

    return True
