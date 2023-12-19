import os
import sys
from logging import Logger
from typing import Callable, Dict, Iterable, List, Union, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

MIL_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(MIL_DIR)
ROOT_DIR = os.path.dirname(EXP_DIR)

MIL_DATA_DIR = os.path.join(MIL_DIR, "data")
DATA_DIR = os.path.join(ROOT_DIR, "data")

sys.path.append(ROOT_DIR)
from core.metric import Metric
from core.datasets import MultimodalMILDataset


class MILModelEvaluator:
    def __init__(self, model: torch.nn.Module, logger: Logger = Logger(__name__)) -> None:
        self.model = model.eval()
        self.logger = logger

    def make_pred_values(
        self, dataloader: torch.utils.data.DataLoader
    ) -> Tuple[Union[np.ndarray, List[np.ndarray]]]:
        y_trues = np.zeros(shape=(len(dataloader)), dtype=np.float32)
        y_probs = np.zeros(shape=(len(dataloader)), dtype=np.float32)
        instance_logits = []
        instance_labels = []
        for idx, batch in enumerate(dataloader):
            snv, instance_label, bag_label = batch

            bag_label = bag_label.squeeze()  # (N, )
            bag_logit, instance_logit = self.model(snv)

            y_trues[idx] = bag_label.item()
            y_probs[idx] = torch.sigmoid(bag_logit).item()

            instance_logits.append(instance_logit.detach().cpu().numpy())
            instance_labels.append(instance_label.squeeze().detach().cpu().numpy())

        return y_trues, y_probs, instance_logits, instance_labels

    def get_optimal_threshold(self, dataloader: torch.utils.data.DataLoader, interval=0.025):
        y_trues, y_probs, _, _ = self.make_pred_values(dataloader)
        n_threds = int(1 / interval)
        thresholds = np.linspace(0, 1, n_threds + 1)
        sensitivities = np.ones(len(thresholds))
        specificities = np.ones(len(thresholds))
        f1_scores = np.ones(len(thresholds))
        for idx, thred in enumerate(thresholds):
            y_preds = y_probs >= thred
            tn, fp, fn, tp = confusion_matrix(y_trues, y_preds).ravel()

            f1_scores[idx] = f1_score(y_trues, y_preds)
            sensitivities[idx] = tp / (tp + fn)
            specificities[idx] = tn / (tn + fp)

        if self.logger is not None:
            self.logger.debug("roc_auc_score:" + str(roc_auc_score(y_trues, y_probs)))
            self.logger.debug("sensitivities by threshold:" + str(sensitivities))
            self.logger.debug("specificities by threshold:" + str(specificities))
            self.logger.debug("f1_scores by threshold:" + str(f1_scores))

        return thresholds[np.argmax(sensitivities + specificities)]

    def evaluate_model(
        self,
        dataset: MultimodalMILDataset,
        metric: Iterable[str],
        threshold=0.5,
        topk=5,
        model_weight=None,
    ) -> Dict[str, float]:
        """Evaluate a model's performance using specified metrics.

        Note:
            This method evaluates a given model's performance by computing various metrics
            based on the provided ground truth labels and predicted probabilities or logits.

        Args:
            dataset (tMultimodalMILDataset): dataset
            metric (Iterable[str]): A list of metric names.
            model (_type_): The model to be evaluated.
            threshold (float, optional): The threshold value for classification. Defaults to 0.5.
            topk (int, optional): The 'k' value for top-k performance metrics. Defaults to 5.
            model_weight (_type_, optional): The weights of the model parameters. Defaults to None.

        Returns:
            Dict[str, float]: A dictionary containing the evaluated metric values.

        Examples:
            >>> mil_evaluator = MILModelEvaluator(test_dataset, logger)
            >>> fold_metric_result = mil_evaluator.evaluate_model(
                    {"f1", "sensitivity", "specificity", "topk_recall"},
                    mil_trainer.model,
                    threshold,
                    self.topk,
                    best_weight,
                )
            >>> print(fold_metric_result)
            metric = {
                "f1": 0.0,
                "sensitivity": 0.0,
                "specificity": 0.0,
                "top5_recall": 0.0,
                "top5_recall_all": 0.0,
            }
        """
        self.model.load_state_dict(model_weight) if model_weight else None

        dataloader = torch.utils.data.DataLoader(dataset, shuffle=False)
        y_trues, y_probs, instance_logits, instance_labels = self.make_pred_values(dataloader)

        is_cnv_confirm = [sum(p_data.cnv_data.y) > 0 for p_data in dataset.patient_datasets.data]

        metric_dict = dict()
        for metric_name in metric:
            metric_func: Callable = getattr(Metric, metric_name)

            if "topk" in metric_name:
                metric_dict[f"topk_any_{topk}"] = metric_func(
                    y_trues, instance_logits, instance_labels, topk, is_any=True
                )
                metric_dict[f"topk_all_{topk}"] = metric_func(
                    y_trues, instance_logits, instance_labels, topk, is_any=False
                )
                # TODO cnv
                instance_logits_cnv = [
                    instance_logit
                    for i, instance_logit in enumerate(instance_logits)
                    if is_cnv_confirm[i]
                ]
                instance_labels_cnv = [
                    instance_label
                    for i, instance_label in enumerate(instance_labels)
                    if is_cnv_confirm[i]
                ]
                metric_dict[f"topk_any_{topk}_cnv"] = metric_func(
                    y_trues[is_cnv_confirm],
                    instance_logits_cnv,
                    instance_labels_cnv,
                    topk,
                    is_any=True,
                )
                metric_dict[f"topk_all_{topk}_cnv"] = metric_func(
                    y_trues[is_cnv_confirm],
                    instance_logits_cnv,
                    instance_labels_cnv,
                    topk,
                    is_any=False,
                )

            else:
                metric_dict[metric_name] = metric_func(y_trues, y_probs, threshold)

        return metric_dict
