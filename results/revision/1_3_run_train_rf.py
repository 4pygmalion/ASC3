"""TabNet 인코더를 이용한 훈련코드

Example:
    $ python3 results/revision/1_3_run_train_rf.py \
        -d /data2/heon_dev/repository/3ASC-Confirmed-variant-Resys/dataset_positive_negative.pickle \
        --run_name RF_MIL_MAX_POOLING \
        --n_epochs 100 \
        --instance_loss_weight 10 \
        --device 3

"""

import os
import sys
import copy
import math
import pickle
import argparse
from typing import List, Tuple
from collections import defaultdict

import torch
import mlflow
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier


REV_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.dirname(REV_DIR)
ROOT_DIR = os.path.dirname(RES_DIR)
EXP_DIR = os.path.join(ROOT_DIR, "experiments")
CORE_DIR = os.path.join(ROOT_DIR, "core")

sys.path.append(ROOT_DIR)
from utils.log_ops import get_logger
from utils.serialization_ops import save_pickle
from core.metric import (
    Metric,
    plot_auroc,
    plot_topk,
    plot_cv_auroc,
    plot_cv_topk,
)
from core.evaluation import MILModelEvaluator
from core.datasets import ExSCNVDataset
from core.data_model import PatientDataSet
from core.trainer import MILTrainer
from core.networks import MultimodalAttentionMIL
from results.revision.revision_utils import TRACKING_URI, EXP_NAME_MIL

RUN_NAME = "baseline"
RANDOM_STATE = 20230524


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_path", help="Path to the data (.pickle) [required]", required=True
    )
    parser.add_argument("--run_name", help="mlflow run name", type=str)
    parser.add_argument(
        "--n_epochs",
        default=100,
        type=int,
        help="Number of epochs for training (default: 100)",
    )
    parser.add_argument(
        "--n_patiences",
        default=15,
        help="Number of patience epochs for early stopping (default: 15)",
    )
    parser.add_argument("--n_hiddens", default=128, help="n dim of hidden vector")
    parser.add_argument(
        "--n_att_hiddens", default=128, help="n dim of attention hidden vector"
    )

    parser.add_argument(
        "--instance_loss_weight",
        default=5000,
        type=int,
        help="Instance loss weight (default: 5000)",
    )

    parser.add_argument(
        "--learning_rate", default=0.001, help="Learning rate (default: 0.001)"
    )
    parser.add_argument("--device", default=-1, help="GPU number")
    parser.add_argument(
        "--base_features",
        default=[
            "ACMG_bayesian",
            "symptom_similarity",
            "vcf_info_QUAL",
            "inhouse_freq",
            "vaf",
            "is_incomplete_zygosity",
        ],
    )
    parser.add_argument(
        "--additional_features",
        default=[
            "gnomad_gene:pLI",
            "gnomad_gene:loeuf",
            "SPLICEAI-VALUE",
            "wes_AC",
            "wgs_AC",
            "clinvar_variant:scv:pathogenicity_n_p",
            "clinvar_variant:scv:pathogenicity_n_b",
            "PVS1",
            "PS1",
            "PS2",
            "PS3",
            "PS4",
            "PM1",
            "PM2",
            "PM3",
            "PM4",
            "PM6",
            "PM5",
            "PP1",
            "PP2",
            "PP3",
            "PP4",
            "PP5",
            "BA1",
            "BS1",
            "BS2",
            "BS3",
            "BS4",
            "BP1",
            "BP2",
            "BP3",
            "BP4",
            "BP5",
            "BP6",
            "BP7",
        ],
    )
    return parser.parse_args()


def post_args_init(args: argparse.Namespace) -> None:
    if args.device == -1:
        args.device = "cpu"
    else:
        setattr(args, "device", f"cuda:{args.device}")

    return


def select_sample_dataset(
    sample_ids: List[str], dataset: PatientDataSet
) -> Tuple[PatientDataSet, PatientDataSet]:
    """지정된 환자 데이터셋에서 주어진 샘플 ID를 기반으로 샘플을 선택하여 데이터셋을 반환
    데이터셋과 나머지 샘플을 포함하는 두 개의 별도 데이터셋을 반환

    Args:
        sample_ids (List[str]): 선택할 샘플 ID가 담긴 리스트
        dataset (PatientDataSet): 전체 환자 데이터셋

    Returns:
        Tuple[PatientDataSet, PatientDataSet]: 두 개의 PatientDataSet 인스턴스로 구성된 튜플
        첫 번째 데이터셋에는 주어진 샘플 ID를 기준으로 선택된 샘플이 포함
        두 번째 데이터셋에는 그외 나머지 샘플이 포함

    Example:
        >>> sample_ids = ["샘플1", "샘플3"]
        >>> dataset = PatientDataSet([...])
        >>> selected_patient_dataset, ramnent_pateint_dataset = select_sample_dataset(
                sample_ids,
                dataset
            )
    """
    mask = np.zeros(shape=(len(dataset)), dtype=np.bool_)
    for idx, patient_data in enumerate(dataset):
        if patient_data.sample_id not in sample_ids:
            continue

        mask[idx] = True

    selected_dataset = dataset[np.where(mask)[0].tolist()]
    ramnent_dataset = dataset[np.where(~mask)[0].tolist()]

    return (selected_dataset, ramnent_dataset)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def save_plot_and_clear(filename):
    """현재 플롯팅된 Figure을 저장하고, Mlflow로깅후 Clear"""
    plt.savefig(filename)
    mlflow.log_artifact(filename)
    os.remove(filename)
    plt.clf()


def snv_cnv_xy_split(patient_dataset: ExSCNVDataset):
    # snv_X_train (List[np.ndarray]) N * (snv_k, snv_feat_x)  N: patients, snv_k: variants
    # snv_y_train = N * (snv_k)
    # cnv_X_train = N * (cnv_k, cnv_feat_x)  N: patients, cnv_k: variants
    # cnv_y_train = N * (cnv_k)
    # bags_train = (N)
    snv_X_patients = []
    snv_y_patients = []
    cnv_X_patients = []
    cnv_y_patients = []
    bags_patients = []
    for patient_idx in range(len(patient_dataset)):
        (snv_x, cnv_x), instance_label, bag_label = patient_dataset[patient_idx]
        snv_x, cnv_x = snv_x.cpu().numpy(), cnv_x.cpu().numpy()
        instance_label = instance_label.cpu().numpy()
        bag_label = bag_label.cpu().numpy()
        snv_X_patients.append(snv_x)
        cnv_X_patients.append(cnv_x)
        snv_num = snv_x.shape[0]
        snv_y = instance_label[0:snv_num]
        cnv_y = instance_label[snv_num:]
        snv_y_patients.append(snv_y)
        cnv_y_patients.append(cnv_y)
        bags_patients.append(bag_label)

    return snv_X_patients, snv_y_patients, cnv_X_patients, cnv_y_patients, bags_patients


if __name__ == "__main__":
    args = get_args()
    post_args_init(args)

    logger = get_logger(__name__)

    with open(args.data_path, "rb") as fh:
        dataset: PatientDataSet = pickle.load(fh)

    with open(os.path.join(EXP_DIR, "external_test_snv_sample_ids.txt")) as fh:
        external_snv_sample_ids = [line.strip() for line in fh.readlines()]
    with open(os.path.join(EXP_DIR, "external_test_cnv_sample_ids.txt")) as fh:
        external_cnv_sample_ids = [line.strip() for line in fh.readlines()]

    ext_snv_patient_dataset, train_val_test_dataset = select_sample_dataset(
        external_snv_sample_ids, dataset
    )
    ext_cnv_patient_dataset, train_val_test_dataset = select_sample_dataset(
        external_cnv_sample_ids, train_val_test_dataset
    )

    # MLFLOW VARS
    mlflow.set_tracking_uri(TRACKING_URI)
    MLFLOW_CLIENT = mlflow.MlflowClient(tracking_uri=TRACKING_URI)
    exp = MLFLOW_CLIENT.get_experiment_by_name(EXP_NAME_MIL)
    if exp is None:
        exp = MLFLOW_CLIENT.create_experiment(EXP_NAME_MIL)
    exp_id = exp.experiment_id
    RUN_NAME = args.run_name if hasattr(args, "run_name") else RUN_NAME

    # RUN
    with mlflow.start_run(experiment_id=exp_id, run_name=RUN_NAME) as parent_run:
        mlflow_np_dataset = mlflow.data.from_numpy(np.array(dataset.data))
        mlflow.log_input(mlflow_np_dataset, context="entire")
        mlflow.log_artifact(os.path.abspath(__file__))
        mlflow.log_artifact(os.path.join(CORE_DIR, "networks.py"))
        mlflow.log_artifact(os.path.join(CORE_DIR, "trainer.py"))
        mlflow.log_params(vars(args))

        stratified_kfold = StratifiedKFold(5, random_state=RANDOM_STATE, shuffle=True)
        stratified_kflod_iter = stratified_kfold.split(
            train_val_test_dataset, train_val_test_dataset.bag_labels
        )

        test_ids = list()
        fold_bag_y_trues = list()
        fold_bag_y_probs = list()
        fold_instance_y_trues = list()
        fold_instance_y_probs = list()
        fold_positive_instance_y_trues = list()
        fold_positive_instance_y_probs = list()
        for fold, (train_val_indices, test_indice) in enumerate(
            stratified_kflod_iter, start=1
        ):
            with mlflow.start_run(
                experiment_id=exp_id, run_name=RUN_NAME + str(fold), nested=True
            ):
                # train_indices, val_indices = train_test_split(
                #     np.arange(len(train_val_indices)),
                #     test_size=0.11,
                #     random_state=RANDOM_STATE,
                #     stratify=train_val_test_dataset.bag_labels[train_val_indices],
                # )
                train_indices = train_val_indices  # RF doesn't need validation set

                snv_scaler = StandardScaler()
                cnv_scaler = StandardScaler()
                train_p_dataset = train_val_test_dataset[train_indices]
                train_dataset = ExSCNVDataset(
                    train_p_dataset,
                    base_features=args.base_features,
                    additional_features=args.additional_features,
                    device=args.device,
                )

                test_dataset = ExSCNVDataset(
                    train_val_test_dataset[test_indice],
                    base_features=args.base_features,
                    additional_features=args.additional_features,
                    device=args.device,
                )
                test_ids.append(
                    [
                        patient_data.sample_id
                        for patient_data in train_val_test_dataset[test_indice]
                    ]
                )

                # fold train and test eval
                snv_model = RandomForestClassifier()
                cnv_model = RandomForestClassifier()
                snv_X_train, snv_y_train, cnv_X_train, cnv_y_train, bags_train = (
                    snv_cnv_xy_split(train_dataset)
                )

                snv_X_all = np.concatenate(snv_X_train)  # (-1, snv_feat_num)
                snv_y_all = np.concatenate(snv_y_train)
                cnv_X_all = np.concatenate(cnv_X_train)  # (-1, cnv_feat_num)
                cnv_y_all = np.concatenate(cnv_y_train)

                snv_model.fit(snv_X_all, snv_y_all)
                cnv_model.fit(cnv_X_all, cnv_y_all)

                snv_X_test, snv_y_test, cnv_X_test, cnv_y_test, bags_test = (
                    snv_cnv_xy_split(test_dataset)
                )

                bag_labels = []
                bag_probs = []
                instance_labels = []
                instance_preds = []
                for snv_X, snv_y, cnv_X, cnv_y, bag in zip(
                    snv_X_test, snv_y_test, cnv_X_test, cnv_y_test, bags_test
                ):  # one for each patient
                    snv_preds = snv_model.predict_proba(snv_X)[:, 1]  # shape = (snv_k)
                    cnv_preds = cnv_model.predict_proba(cnv_X)[:, 1]  # shape = (cnv_k)
                    preds = np.concatenate([snv_preds, cnv_preds])
                    labels = np.concatenate([snv_y, cnv_y])
                    if bag:  # positive only for top-k recall
                        instance_preds.append(preds)
                        instance_labels.append(labels)

                    bag_probs.append(np.max(preds))
                    bag_labels.append(bag)

                fold_bag_y_trues.append(bag_labels)
                fold_bag_y_probs.append(bag_probs)
                fold_positive_instance_y_trues.append(instance_labels)
                fold_positive_instance_y_probs.append(instance_preds)

                fig, axes = plot_auroc(bag_labels, bag_probs)
                save_plot_and_clear("auroc.png")

                fold_bag_y_trues.append(bag_labels)
                fold_bag_y_probs.append(bag_probs)

                plot_topk(instance_labels, instance_preds)
                save_plot_and_clear("topk.png")

                fold_instance_y_trues.append(instance_labels)
                fold_instance_y_probs.append(instance_preds)

                mlflow.log_metric(
                    "test_positive_top5",
                    Metric.topk_recall(
                        bag_labels,
                        instance_preds,
                        instance_labels,
                        k=5,
                        is_any=True,
                    ),
                )
                plot_topk(instance_labels, instance_preds)
                save_plot_and_clear("test_positive_topk.png")
                fold_positive_instance_y_trues.append(instance_labels)
                fold_positive_instance_y_probs.append(instance_preds)

        plot_cv_auroc(fold_bag_y_trues, fold_bag_y_probs)
        save_plot_and_clear("cv_auroc.png")

        plot_cv_topk(fold_positive_instance_y_trues, fold_positive_instance_y_probs)
        save_plot_and_clear("cv_topk.png")

        save_pickle(
            {
                "fold_bag_y_trues": fold_bag_y_trues,
                "fold_bag_y_probs": fold_bag_y_probs,
                "fold_instance_y_trues": fold_instance_y_trues,
                "fold_instance_y_probs": fold_instance_y_probs,
                "fold_positive_instance_y_trues": fold_positive_instance_y_trues,
                "fold_positive_instance_y_probs": fold_positive_instance_y_probs,
                "test_ids": test_ids,
            },
            path="fold_result.pickle",
        )
        mlflow.log_artifact("fold_result.pickle")
        mlflow.log_artifact(args.data_path)

        # ext_snv_dataset = ExSCNVDataset(
        #     ext_snv_patient_dataset,
        #     base_features=args.base_features,
        #     additional_features=args.additional_features,
        #     scalers=scalers,
        #     device=args.device,
        # )
        # ext_cnv_dataset = ExSCNVDataset(
        #     ext_cnv_patient_dataset,
        #     base_features=args.base_features,
        #     additional_features=args.additional_features,
        #     scalers=scalers,
        #     device=args.device,
        # )
        # train_data_loader = torch.utils.data.DataLoader(
        #     train_dataset, shuffle=True
        # )
        # val_data_loader = torch.utils.data.DataLoader(val_dataset, shuffle=True)

        # model = MultimodalAttentionMIL(
        #     n_snv_features=len(train_dataset.feature_indices),
        #     n_cnv_features=3,
        #     n_hiddens=args.n_hiddens,
        #     n_att_hiddens=args.n_att_hiddens,
        # ).to(args.device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, verbose=True
        # )
        # trainer = MILTrainer(
        #     model,
        #     optimizer,
        #     bag_loss_weight=1,
        #     instance_loss_weight=args.instance_loss_weight,
        #     device=args.device,
        # )

        # patience = 0
        # # best_loss: float = math.inf
        # best_metric = -math.inf
        # for epoch in range(1, args.n_epochs + 1):
        #     (
        #         train_loss,
        #         train_acc,
        #         train_top5,
        #         train_cnv_top5,
        #     ) = trainer.run_epoch(
        #         phase="train", epoch=epoch, dataloader=train_data_loader
        #     )
        #     val_loss, val_acc, val_top5, val_cnv_top5 = trainer.run_epoch(
        #         phase="val", epoch=epoch, dataloader=val_data_loader
        #     )
        #     scheduler.step(val_loss)

        #     mlflow.log_metrics(
        #         {
        #             "train_loss": train_loss,
        #             "train_acc": train_acc,
        #             "train_topk_any": train_top5,
        #             "train_topk_cnv": train_cnv_top5,
        #             "val_loss": val_loss,
        #             "val_bag_acc": val_acc,
        #             "val_topk_any": val_top5,
        #             "val_topk_cnv": val_cnv_top5,
        #         },
        #         step=epoch,
        #     )

        #     if best_metric < val_top5 + val_cnv_top5:
        #         best_metric = val_top5 + val_cnv_top5
        #         best_weight = copy.deepcopy(model.state_dict())
        #         patience = 0

        #     else:
        #         patience += 1
        #         if patience == args.n_patiences:
        #             break

        # model.load_state_dict(best_weight)
        # evaluator = MILModelEvaluator(model, logger)
        # threshold = evaluator.get_optimal_threshold(val_data_loader)

        # fold_metric_result = evaluator.evaluate_model(
        #     test_dataset,
        #     Metric.available_metrics,
        #     threshold,
        # )
        # (
        #     bag_labels,
        #     bag_probs,
        #     instance_logits,
        #     instance_labels,
        # ) = evaluator.make_pred_values(
        #     torch.utils.data.DataLoader(test_dataset)
        # )

        # ext_snv_result = evaluator.evaluate_model(
        #     ext_snv_dataset,
        #     ["topk_recall"],
        #     threshold,
        # )
        # mlflow.log_metrics({"topk_any_5.ext.snv": ext_snv_result["topk_any_5"]})
        # logger.info(f"topk_any_5.ext.snv: {ext_snv_result['topk_any_5']}")

        # ext_cnv_result = evaluator.evaluate_model(
        #     ext_cnv_dataset,
        #     ["topk_recall"],
        #     threshold,
        # )
        # mlflow.log_metrics({"topk_any_5.ext.cnv": ext_cnv_result["topk_any_5"]})
        # logger.info(f"topk_any_5.ext.cnv: {ext_cnv_result['topk_any_5']}")

        # logger.info("logging checkpoint into MLflow")
        # mlflow.pytorch.log_model(model, "checkpoint")

        # logger.info("logging scaler into MLflow")
        # torch.save({"scaler": scalers}, "scaler.pt")
        # mlflow.log_artifact("scaler.pt", "checkpoint")
        # os.remove("scaler.pt")
