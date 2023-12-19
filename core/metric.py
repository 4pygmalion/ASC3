import math
from itertools import cycle
from typing import Iterable, Tuple, List
from dataclasses import dataclass, asdict

import torch
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix, roc_curve, auc


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@dataclass
class MetricHolder:
    """Metric들을 저장하기위한 플레이스홀더

    Example:
    >>> @dataclass
    >>> class TestMetricHolder(MetricHolder):
    >>>     f1: AverageMeter = AverageMeter()
    >>>     accuracy: AverageMeter = AverageMeter()

    >>> metric_holder = TestMetricHolder()
    >>> metric_holder.update(
            {
                "f1": 0.80,
                "accuracy": 90,
            },
            n=10
        )
    >>> metric_holder.to_dict()
    {"f1": 0.80, "accuracy": 90}
    """

    def update(self, metrics: dict, n: int):
        for name, value in metrics.items():
            if hasattr(self, name):
                average_meter = getattr(self, name)
                average_meter.update(value, n)

            else:
                raise ValueError(f"Not defined class variable({name}).")

    def to_dict(self, prefix: str = str()):
        if not prefix:
            return {attr: round(meter.avg, 5) for attr, meter in asdict(self).items()}

        return {f"{prefix}_{attr}": round(meter.avg, 5) for attr, meter in asdict(self).items()}


@dataclass
class MILMetricHolder(MetricHolder):
    loss: AverageMeter = AverageMeter()
    f1: AverageMeter = AverageMeter()
    accuracy: AverageMeter = AverageMeter()
    snv_cnv_top5: AverageMeter = AverageMeter()
    cnv_top5: AverageMeter = AverageMeter()


def topk_recall(
    instance_prob: np.ndarray, instance_label: np.ndarray, k: int = 5, is_any: bool = True
) -> bool:
    """주어진 인스턴스 확률과 라벨에 대해 top-k 재현율을 계산

    Args:
        instance_prob (np.ndarray): 인스턴스 확률
        instance_label (np.ndarray): 인스턴스에 대한 라벨의 바이너리(이진) 형태
        k (int, optional): top-k 재현율 계산에 사용되는 k 값
        is_any (bool, optional): 적어도 하나의 변이가 hit되어도 hit으로 구분할 것인지여부

    Returns:
        bool: hit 여부
    """

    instance_label = instance_label.ravel() if instance_label.ndim >= 2 else instance_label
    instance_prob = instance_prob.ravel() if instance_prob.ndim >= 2 else instance_prob

    if len(instance_label) != len(instance_prob):
        raise ValueError("instance label length is not equal with that of instance_prob")

    label_indices, *_ = np.where(instance_label == 1)
    prediction_indices = np.argsort(instance_prob)[-k:]

    intersection = np.intersect1d(label_indices, prediction_indices)

    if len(intersection) == 0:
        return False

    if is_any:
        return True

    if len(label_indices) == len(intersection):
        return True

    return False


class Metric:
    available_metrics = {"f1", "sensitivity", "specificity", "topk_recall"}

    @staticmethod
    def _calculate_conf_mat(y_trues, y_probs, threshold):
        cm = confusion_matrix(y_trues, y_probs >= threshold)
        tn, fp, fn, tp = cm.ravel()
        return tn, fp, fn, tp

    @classmethod
    def f1(cls, y_trues, y_probs, threshold):
        tn, fp, fn, tp = cls._calculate_conf_mat(y_trues, y_probs, threshold)
        return 2 * tp / (2 * tp + fp + fn)

    @classmethod
    def sensitivity(cls, y_trues, y_probs, threshold):
        tn, fp, fn, tp = cls._calculate_conf_mat(y_trues, y_probs, threshold)
        return tp / (tp + fn)

    @classmethod
    def specificity(cls, y_trues, y_probs, threshold):
        tn, fp, fn, tp = cls._calculate_conf_mat(y_trues, y_probs, threshold)
        return tn / (tn + fp)

    @classmethod
    def topk_recall(
        cls,
        bag_labels: torch.Tensor,
        instance_probs: Iterable[np.ndarray],
        instance_labels: Iterable[np.ndarray],
        k: int,
        is_any: bool,
    ) -> float:
        """TopK recall을 계산

        Args:
            bag_labels (torch.Tensor): Bag labels(환자 라벨)
            instance_probs (Iterable[np.ndarray]): 환자의 변이의 원인변이 확률
            instance_labels (Iterable[np.ndarray]): 환자의 변이의 원인변이 라벨
            k (int): at K
            is_any (bool): 하나의 변이라도 맞추는 경우 hit으로 볼 것인지 여부

        Returns:
            float: average top k recall

        Example:
            >>> mil_evaluator = MILModelEvaluator(test_dataset, logger)
            >>> y_trues, y_probs, instance_logits, instance_labels = self.make_pred_values(model)
            >>> Metric.topk_recall(
                    y_trues,
                    instance_logits,
                    instance_labels,
                    k=topk,
                    is_any=False
                )
        """
        num_neg_sample = 0
        topks = 0
        for bag_label, instance_prob, instance_label in zip(
            bag_labels, instance_probs, instance_labels
        ):
            if not bag_label:
                num_neg_sample += 1
                continue

            is_hit = topk_recall(
                instance_prob,
                instance_label,
                k,
                is_any,
            )
            if is_hit:
                topks += 1

        total_n = len(bag_labels) - num_neg_sample
        if total_n == 0:
            return math.nan

        return topks / total_n


def plot_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[plt.Figure, plt.Axes]:
    """그림을 사용하여 AUROC (Area Under the Receiver Operating Characteristic Curve)를 시각화

    Args:
        y_true (np.ndarray): 실제 라벨 값의 배열.
        y_prob (np.ndarray): 모델의 예측 확률 값의 배열.

    Returns:
        Tuple[plt.Figure, plt.Axes]: Matplotlib의 Figure와 Axes 객체를 반환

    Example:
        >>> fig, axes = plot_auroc(bag_labels, bag_probs)
        >>> plt.savefig("auroc.png")
        >>> mlflow.log_artifact("auroc.png")
        >>> os.remove("auroc.png")
        >>> plt.clf()
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, axes = plt.subplots()
    axes.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUROC = {roc_auc:.3f}")
    axes.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axes.set_xlim([0.0, 1.0])
    axes.set_ylim([0.0, 1.05])
    axes.set_xlabel("False Positive Rate")
    axes.set_ylabel("True Positive Rate")
    axes.set_title("Receiver Operating Characteristic (ROC)")
    axes.legend(loc="lower right")

    return fig, axes


def plot_cv_auroc(
    fold_y_trues: List[np.ndarray],
    fold_y_probs: List[np.ndarray],
) -> None:
    """CV결과를 담은 AUROC를 시각화

    Args:
        y_true (List[np.ndarray]): 실제 라벨 값의 배열.
        y_prob (List[np.ndarray]): 모델의 예측 확률 값의 배열.

    Returns:
        None: Matplotlib의 Figure와 Axes 객체를 반환

    Example:
        >>> fold_bag_y_trues = list()
        >>> fold_bag_y_probs = list()
        >>> for fold in ...
                ...
        >>>     fold_bag_y_trues.append(bag_labels)
        >>>     fold_bag_y_probs.append(bag_probs)

        >>> plot_cv_auroc(fold_bag_y_trues, fold_bag_y_probs)
        >>> plt.savefig("cv_auroc.png")
        >>> mlflow.log_artifact("cv_auroc.png")
        >>> os.remove("cv_auroc.png")
        >>> plt.clf()
    """
    plt.figure(figsize=(8, 6))
    lw = 2
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "red", "purple"])

    # Initialize lists to store individual fold's FPR, TPR, and AUROC
    all_fpr = []
    all_tpr = []
    all_roc_auc = []

    # Calculate AUROC for each fold and plot the ROC curve
    for i, (y_true, y_prob) in enumerate(zip(fold_y_trues, fold_y_probs)):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        all_roc_auc.append(roc_auc)

        plt.plot(
            fpr,
            tpr,
            color=next(colors),
            lw=lw,
            label="ROC curve (fold %d) (area = %0.2f)" % (i + 1, roc_auc),
        )

    # Calculate the mean FPR and TPR across folds
    max_length = max(len(arr) for arr in all_fpr)
    interp_fpr = [
        interp1d(np.arange(len(fpr)), fpr)(np.linspace(0, len(fpr) - 1, max_length))
        for fpr in all_fpr
    ]
    interp_tpr = [
        interp1d(np.arange(len(tpr)), tpr)(np.linspace(0, len(tpr) - 1, max_length))
        for tpr in all_tpr
    ]
    mean_fpr = np.mean(interp_fpr, axis=0)
    mean_tpr = np.mean(interp_tpr, axis=0)
    mean_roc_auc = auc(mean_fpr, mean_tpr)

    # Average ROC curve
    plt.plot(
        mean_fpr,
        mean_tpr,
        color="navy",
        linestyle="--",
        label="Mean ROC curve (area = %0.2f)" % mean_roc_auc,
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")


def plot_topk(
    instance_labels: List[np.ndarray],
    instance_probs: List[np.ndarray],
    ks: List[int] = [1, 2, 3, 4, 5, 10, 15, 20, 100],
):
    """
    주어진 instance_labels와 instance_probs를 기반으로 top-k recall을 계산하고,
    다양한 k 값에 대한 평균 recall을 시각화

    Args:
        instance_labels (List[np.ndarray]): 각 인스턴스의 실제 라벨을 포함한 리스트.
        instance_probs (List[np.ndarray]): 각 인스턴스의 모델 예측 확률을 포함한 리스트.
        ks (List[int], optional): 시각화할 top-k 값들의 리스트.
            기본값은 [1, 2, 3, 4, 5, 10, 15, 20, 100].

    Returns:
        Tuple[plt.Figure, plt.Axes]: Matplotlib의 Figure와 Axes 객체를 반환

    Example:
        >>> plot_topk(
                instance_labels,
                [sigmoid(logit) for logit in instance_logits],
            )
        >>> plt.savefig("topk.png")
        >>> mlflow.log_artifact("topk.png")
        >>> os.remove("topk.png")
        >>> plt.clf()
    """
    if len(instance_labels) != len(instance_probs):
        msg = (
            f"length of y_labels (n={len(instance_labels)}) is not equal "
            f"to y_probs(n={len(instance_probs)})"
        )
        raise ValueError(msg)

    ys = list()
    for k in ks:
        k_avg = list()
        for y_label, y_prob in zip(instance_labels, instance_probs):
            if sum(y_label) == 0:
                continue

            k_avg.append(topk_recall(y_prob, y_label, k=k))

        ys.append(sum(k_avg) / len(k_avg))

    fig, axes = plt.subplots()
    sns.barplot(
        data=pd.DataFrame({"topk": ks, "recall": ys}), x="topk", y="recall", color="gray", ax=axes
    )
    return fig, axes


def plot_cv_topk(
    fold_instance_labels: List[List[np.ndarray]],
    fold_instance_probs: List[List[np.ndarray]],
    ks: List[int] = [1, 2, 3, 4, 5, 10, 15, 20, 100],
):
    """
    CV결과에서 instance_labels와 instance_probs를 기반으로 top-k recall을 계산하고,
    다양한 k 값에 대한 평균 recall을 시각화

    Args:
        instance_labels (List[List[np.ndarray]]): fold 별, 각 인스턴스의 실제 라벨을 포함한 리스트.
        instance_probs (List[List[np.ndarray]]): 각 인스턴스의 모델 예측 확률을 포함한 리스트.
        ks (List[int], optional): 시각화할 top-k 값들의 리스트.
            기본값은 [1, 2, 3, 4, 5, 10, 15, 20, 100].

    Returns:
        Tuple[plt.Figure, plt.Axes]: Matplotlib의 Figure와 Axes 객체를 반환
    """
    if len(fold_instance_labels) != len(fold_instance_probs):
        msg = (
            f"length of y_labels (n={len(fold_instance_labels)}) is not equal "
            f"to y_probs(n={len(fold_instance_probs)})"
        )
        raise ValueError(msg)

    data = list()

    for fold, (instance_labels, instance_probs) in enumerate(
        zip(fold_instance_labels, fold_instance_probs), start=1
    ):
        for k in ks:
            k_avg = list()
            for y_label, y_prob in zip(instance_labels, instance_probs):
                if sum(y_label) == 0:
                    continue

                k_avg.append(topk_recall(y_prob, y_label, k=k))
            data.append([fold, k, sum(k_avg) / len(k_avg)])

    df = pd.DataFrame(data, columns=["fold", "topk", "recall"])

    fig, axes = plt.subplots()
    sns.barplot(data=df, x="topk", y="recall", color="gray", ax=axes)

    return fig, axes
