import os
import sys
import copy
from typing import Tuple, Dict, Any

import torch
import numpy as np
import logging
from progress.bar import Bar

MIL_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(MIL_DIR)
ROOT_DIR = os.path.dirname(EXP_DIR)
MIL_DATA_DIR = os.path.join(MIL_DIR, "data")
DATA_DIR = os.path.join(ROOT_DIR, "data")

sys.path.append(ROOT_DIR)
from core.networks import *
from core.losses import focal_with_bce, focal_loss, pointwise_ranknet_loss
from core.metric import topk_recall, AverageMeter


class MILTrainer:
    def __init__(
        self,
        model: torch.nn.modules.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        bag_loss_weight: float = 1,
        instance_loss_weight: float = 1,
        run_id: str = None,
        tracker=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.bag_loss_weight = bag_loss_weight
        self.instance_loss_weight = instance_loss_weight
        self.run_id = run_id
        self.tracker = tracker

    @staticmethod
    def get_acc(label: torch.Tensor, logit: torch.Tensor, cutoff=0.5) -> float:
        """주어진 Ground Truth(label)과 모델의 최종반환(logit)을 이용하여
        model confidnece의 Cutoff을 기준으로 정확도(Accuracy)을 반환

        Args:
            label (torch.Tensor): Ground truth label
            logit (torch.Tensor): model final output (=logit)

        Return
            accuracy (float): 정확도
        """

        prob = torch.sigmoid(logit)
        pred_label = (prob >= cutoff).float()

        n_acc = torch.sum(pred_label == label)
        n_instance = len(label)

        return round(float((n_acc / n_instance).detach().cpu().numpy()), 5)

    def get_best_weight(self):
        return (
            self.best_weight
            if hasattr(self, "best_weight")
            else self.model.state_dict()
        )

    def make_bar_sentence(
        self,
        phase: str,
        epoch: int,
        total_step: int,
        step: int,
        eta: str,
        total_loss: float,
        bag_acc: float,
        top_k_recall: float,
    ) -> str:
        """ProgressBar의 stdout의 string을 생성하여 반환

        Args:
            phase (str): Epoch의 phase
            epoch (int): epoch
            total_step (int): total steps for one epoch
            step (int): Step (in a epoch)
            eta (str): Estimated Time of Arrival
            loss (float): loss
            bag_acc (float): bag accuracy
            top_k_recall (float): top 5 recall (any)

        Returns:
            str: progressbar senetence

        """
        bag_acc = round(bag_acc, 5)
        total_loss = round(total_loss, 5)
        top_k_recall = round(top_k_recall, 5)

        return (
            f"{phase} | EPOCH {epoch}: [{step}/{total_step}] | "
            f"eta:{eta} | total_loss: {total_loss} | bag_acc: {bag_acc} | "
            f"top5recall(any): {top_k_recall}"
        )

    def run_epoch(
        self,
        phase: str,
        epoch: int,
        dataloader: torch.utils.data.DataLoader,
        cnv_loss_weight: float = 1,
    ) -> Tuple[float, float]:
        """1회 Epoch을 각 페이즈(train, validation)에 따라서 학습하거나 손실값을
        반환함.

        Note:
            - 1 epoch = Dataset의 전체를 학습한경우
            - 1 step = epoch을 하기위해 더 작은 단위(batch)로 학습할 떄의 단위

        Args:
            phase (str): training or validation
            epoch (int): epoch
            dataloader (torch.utils.data.DataLoader): dataset (train or validation)

        Returns:
            Tuple: loss, accuracy, top_k_recall
        """
        total_step = len(dataloader)
        bar = Bar(max=total_step)

        total_loss = AverageMeter()
        top_k_recall = AverageMeter()
        avg_bag_acc = AverageMeter()
        top_k_recall_cnv = AverageMeter()

        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits

        for step, batch in enumerate(dataloader):
            x, instance_label, bag_label = batch
            instance_label = instance_label.squeeze(dim=0)  # (N, )
            bag_label = bag_label.squeeze(dim=-1)  # (N, )

            if phase == "train":
                self.model.train()
            else:
                self.model.eval()

            bag_logit, instance_logit = self.model(x)  # (N, )
            instance_prob = torch.sigmoid(instance_logit)

            loss = bce_loss(bag_logit, bag_label)
            loss += bce_loss(instance_logit, instance_label)

            if bag_label == True:
                loss += 0.9 * pointwise_ranknet_loss(
                    instance_prob, instance_label, sigma=1
                )

            is_cnv_confirm = instance_label[-len(x[1].squeeze()) :].sum() > 0
            if cnv_loss_weight != 1 and is_cnv_confirm:
                loss *= cnv_loss_weight

            # mb sgd
            if phase == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # bag
            total_loss.update(loss.item(), 1)
            avg_bag_acc.update(self.get_acc(bag_label, bag_logit), len(bag_logit))
            if bag_label:
                topk_flag: bool = topk_recall(
                    instance_prob.detach().cpu().numpy(),
                    instance_label.detach().cpu().numpy(),
                    k=5,
                )
                top_k_recall.update(topk_flag)
                if is_cnv_confirm:
                    top_k_recall_cnv.update(topk_flag)

            bar.suffix = self.make_bar_sentence(
                phase=phase,
                epoch=epoch,
                step=step,
                total_step=total_step,
                eta=bar.eta,
                total_loss=total_loss.avg,
                bag_acc=avg_bag_acc.avg,
                top_k_recall=top_k_recall.avg,
            )
            bar.next()

            if self.tracker is not None and step % 100 == 99 and phase == "train":
                self.tracker.log_metric(
                    run_id=self.run_id,
                    key="train_loss",
                    value=total_loss.avg,
                    step=epoch,
                )
                self.tracker.log_metric(
                    run_id=self.run_id,
                    key="train_bag_acc",
                    value=avg_bag_acc.avg,
                    step=epoch,
                )
                self.tracker.log_metric(
                    run_id=self.run_id,
                    key="train_topk_any",
                    value=top_k_recall.avg,
                    step=epoch,
                )

        bar.finish()

        return total_loss.avg, avg_bag_acc.avg, top_k_recall.avg, top_k_recall_cnv.avg

    def train_model(
        self,
        n_epochs: int,
        n_patiences: int,
        train_data_loader: torch.utils.data.DataLoader,
        val_data_loader: torch.utils.data.DataLoader,
        logger: logging.Logger = None,
        pretrain_set: Tuple[np.array, np.array] = None,
    ) -> Dict[str, Any]:
        if pretrain_set is not None:
            self.model.pretrain(
                pretrain_set[0],
                pretrain_set[1],
                device=self.device,
            )

        patience = 0
        best_topk: float = 0.0

        for epoch in range(1, n_epochs + 1):
            (
                train_loss,
                train_bag_acc,
                train_top_k_recall,
                train_top_k_recall_cnv,
            ) = self.run_epoch("train", epoch, train_data_loader)
            (
                val_loss,
                val_bag_acc,
                val_top_k_recall,
                val_top_k_recall_cnv,
            ) = self.run_epoch("val", epoch, val_data_loader)

            if logger is not None:
                logger.debug(
                    f"\nepoch: {epoch}, train_loss: {train_loss}, train_acc: {train_bag_acc}"
                )
                logger.debug(
                    f"epoch: {epoch}, val_loss: {val_loss}, val_acc: {val_bag_acc}"
                )
                logger.debug(
                    f"epoch: {epoch}, topk_any_5: {val_top_k_recall}, "
                    + f"topk_all_5: {val_top_k_recall}"
                )

            if self.run_id and self.tracker:
                self.tracker.log_metric(
                    run_id=self.run_id, key="val_loss", value=val_loss, step=epoch
                )
                self.tracker.log_metric(
                    run_id=self.run_id, key="val_bag_acc", value=val_bag_acc, step=epoch
                )
                self.tracker.log_metric(
                    run_id=self.run_id,
                    key="val_topk_any",
                    value=val_top_k_recall,
                    step=epoch,
                )
                self.tracker.log_metric(
                    run_id=self.run_id,
                    key="val_topk_cnv",
                    value=val_top_k_recall_cnv,
                    step=epoch,
                )
                val_topk_mean = (val_top_k_recall + val_top_k_recall_cnv) / 2
                self.tracker.log_metric(
                    run_id=self.run_id,
                    key="val_topk_mean",
                    value=val_topk_mean,
                    step=epoch,
                )

            if best_topk < val_topk_mean:
                best_topk = val_topk_mean
                self.best_weight = copy.deepcopy(self.model.state_dict())
                patience = 0

            else:
                patience += 1
                if patience == n_patiences:
                    break


class TabNetTrainer(MILTrainer):
    """TabNet훈련용 클레스

    Note:
        TabNet은 Ghost normalization 때문에, CNV가 없는 경우(shape=(1, 3))
        forward을 제외하고, 진행함. 이에 따라, losses도 instance label에서도
        CNV을 제외하여야함.
    """

    def __init__(
        self,
        model: torch.nn.modules.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        bag_loss_weight,
        instance_loss_weight,
    ):
        super().__init__(
            model, optimizer, device, bag_loss_weight, instance_loss_weight
        )

    def run_epoch(
        self, phase: str, epoch: int, dataloader: torch.utils.data.DataLoader
    ) -> Tuple[float, float]:
        """1회 Epoch을 각 페이즈(train, validation)에 따라서 학습하거나 손실값을
        반환함.

        Args:
            phase (str): training or validation
            epoch (int): epoch
            dataloader (torch.utils.data.DataLoader): dataset (train or validation)

        Returns:
            Tuple: loss, accuracy, top_k_recall
        """
        total_step = len(dataloader)
        bar = Bar(max=total_step)

        total_loss = AverageMeter()
        top_k_recall_any = AverageMeter()
        avg_bag_acc = AverageMeter()
        for step, batch in enumerate(dataloader):
            (snv, cnv), instance_label, bag_label = batch
            instance_label = instance_label.squeeze(dim=0)  # (N, )
            bag_label = bag_label.squeeze()  # (N, )

            if not bag_label:
                bar.next()
                continue

            if phase == "train":
                self.model.train()
            else:
                self.model.eval()

            if cnv.ndim == 2 and len(cnv) == 1:
                instance_label = instance_label[:-1]
            elif cnv.ndim == 3 and len(cnv[0]) == 1:
                instance_label = instance_label[:-1]

            bag_logit, instance_logit = self.model((snv, cnv))  # (N, )
            bag_logit = bag_logit.squeeze()
            instance_prob = torch.sigmoid(instance_logit)

            loss = focal_with_bce(
                bag_logit,
                bag_label,
                instance_prob,
                instance_label,
                self.bag_loss_weight,
                self.instance_loss_weight,
            )

            if phase == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # bag
            total_loss.update(loss.item(), 1)
            if bag_label:
                top_k_recall_any.update(
                    topk_recall(
                        instance_logit.detach().cpu().numpy(),
                        instance_label.detach().cpu().numpy(),
                        k=5,
                    )
                )

            avg_bag_acc.update(0, 1)
            bar.suffix = self.make_bar_sentence(
                phase=phase,
                epoch=epoch,
                step=step,
                total_step=total_step,
                eta=bar.eta,
                total_loss=total_loss.avg,
                bag_acc=avg_bag_acc.avg,
                top_k_recall=top_k_recall_any.avg,
            )
            bar.next()

        bar.finish()

        return total_loss.avg, avg_bag_acc.avg, top_k_recall_any.avg


class MILTrainerBuilder:
    def __init__(self, logger) -> None:
        self.logger = logger

    def set_run_id(self, run_id):
        self.run_id = run_id
        return self

    def set_tracker(self, tracker):
        self.tracker = tracker
        return self

    def set_n_snv_features(self, n_snv_features):
        self.n_snv_features = n_snv_features
        return self

    def set_n_cnv_features(self, n_cnv_features):
        self.n_cnv_features = n_cnv_features
        return self

    def set_n_hiddens(self, n_hiddens):
        self.n_hiddens = n_hiddens
        return self

    def set_n_att_hiddens(self, n_att_hiddens):
        self.n_att_hiddens = n_att_hiddens
        return self

    def set_device(self, device):
        self.device = device
        return self

    def set_model_class(self, model_class):
        self.model_class = model_class
        return self

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        return self

    def set_alpha(self, alpha):
        self.alpha = alpha
        return self

    def set_gamma(self, gamma):
        self.gamma = gamma
        return self

    def set_bag_loss_weight(self, bag_loss_weight):
        self.bag_loss_weight = bag_loss_weight
        return self

    def set_instance_loss_weight(self, instance_loss_weight):
        self.instance_loss_weight = instance_loss_weight
        return self

    def set_use_sparsemax(self, use_sparsemax):
        self.use_sparsemax = use_sparsemax
        return self

    def build(self) -> MILTrainer:
        self.model = eval(self.model_class)(
            n_snv_features=self.n_snv_features,
            n_cnv_features=self.n_cnv_features,
            n_hiddens=self.n_hiddens,
            n_att_hiddens=self.n_att_hiddens,
            use_sparsemax=self.use_sparsemax,
        ).to(self.device)

        return MILTrainer(
            model=self.model,
            optimizer=torch.optim.Adam(self.model.parameters(), lr=self.learning_rate),
            device=self.device,
            bag_loss_weight=self.bag_loss_weight,
            instance_loss_weight=self.instance_loss_weight,
            alpha=self.alpha,
            gamma=self.gamma,
            run_id=self.run_id,
            tracker=self.tracker,
        )
