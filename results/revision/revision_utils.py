import os
import sys
from typing import List, Union

import tqdm
import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split

REV_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.dirname(REV_DIR)
ROOT_DIR = os.path.dirname(RES_DIR)

sys.path.append(ROOT_DIR)
from core.data_model import PatientDataSet

TRACKING_URI = ""
DATA_PATH = "/data2/heon_dev/repository/3ASC-Confirmed-variant-Resys/dataset_positive_negative.pickle"

ORIGINAL_REPO_EXP_DIR = (
    "/data/heon_dev/repository/3ASC-Confirmed-variant-Resys/experiments"
)
RANDOM_STATE = 20230524
RUN_ASC3_W_RANKNET = "29f1dfc8ee2b464cb6ec36627f12e429"
RUN_ASC3_WO_RANKNET = "87cec01ac82a42a6a6fffce3b7543597"
FOLD_RESULTS_PICKLE = "mlflow-artifacts:/6/{run_id}/artifacts/fold_result.pickle"


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
