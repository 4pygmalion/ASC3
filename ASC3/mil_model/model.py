import os
import sys
import time
from collections import defaultdict
from typing import Tuple, Dict, Any
from logging import Logger

import torch
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from omegaconf import OmegaConf

MIL_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ASC3_DIR = os.path.dirname(MIL_MODEL_DIR)
ROOT_DIR = os.path.dirname(ASC3_DIR)
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "data", "checkpoint")
sys.path.append(ROOT_DIR)

from core.snv_factory import SNVFeaturizer
from core.cnv_factory import CNVFeaturizer
from core.data_model import PatientData, PatientDataSet, SNVData, CNVData, Variant
from core.datasets import ExSCNVDataset
from core.dynamodb_ops import DynamoDBClient
from ASC3.mil_model.data_model import MILRequest, SNVFeature, CNVFeature

from core.networks import MultimodalAttentionMIL

from utils.log_ops import get_logger
from mlflow_settings import TRACKING_URI


class MILPredictor:
    """Variant Recommendation sys with Multiple Instance Learning

    Example:
        >>> predictor = MILPredictor(config)
        >>> patient_data = predictor.build_data_from_file("EPG23-MIBO")
        >>> predictor.predict(patient_data)
        (0.0,
            {
                ('2-216271029-T-C', 'OMIM:601894'): 0.9963311553001404,
                ('2-11925201-A-AG', 'OMIM:268200'): 0.02079474739730358,
                ('7-155599019-T-C', 'OMIM:142945'): 0.003935012500733137,
                ('9-14841598-C-T', 'OMIM:608980'): 0.0021341294050216675,
                ('12-54925197-G-T', 'OMIM:618982'): 0.0020637940615415573,
                ('X-53616801-A-C', 'OMIM:309590'): 0.0017595314420759678,
                ('X-53616801-A-C', 'OMIM:314320'): 0.0015471501974388957,
                ('9-14841598-C-T', 'OMIM:248450'): 0.001166412839666009,
                ('17-76446331-C-T', 'OMIM:618643'): 0.0005031892796978354,
                ('1-155208081-C-T', 'OMIM:231000'): 0.0003688583383336663,
                ('22-51159712-C-T', 'OMIM:606232'): 8.111949864542112e-05,
                ('5-176824007-G-A', 'OMIM:616963'): 1.9060827980865724e-05,
                ('5-176824007-G-A', 'OMIM:613388'): 1.6208001397899352e-05,
            }
        )

        # TODO:
        >>> query = ... # 클라이언트로부터 전달받은 json
        >>> patient_data = predictor.convert_json_to_patientdata(query)
        >>> predictor.predict(patient_data)

    """

    def __init__(
        self, config: dict, device: str = "cpu", logger: Logger = Logger(__name__)
    ) -> None:
        """클래스 생성자

        Args:
            config (dict): 설정 정보가 담긴 딕셔너리
            logger (Logger): 로깅을 위한 Logger 인스턴스
        """
        self.config = config
        self.logger = logger
        self.device = device
        self.trials = 0
        self._set_model()
        self.feature_name = (
            self.config["MIL_MODEL"]["BASE_FEATURE"]
            + self.config["MIL_MODEL"]["ADDITIONAL_FEATURES"]
            + self.config["MIL_MODEL"]["RULES"]
        )

    def _set_featurizer(self) -> None:
        """피처라이저 설정"""

        self.logger.info("Load dynamodb client.")
        self.dynamodb_client = DynamoDBClient(
            os.path.join(ROOT_DIR, self.config["KEYFILE"])
        )

        self.logger.info("Load SNV, CNV featurizer")
        self.snv_featurizer = SNVFeaturizer(
            sequencing="wes",
            snv_root_path=self.config["MIL_MODEL"]["SNV_FEATERIZER"]["snv_root_path"],
            inhouse_freq_path=os.path.join(
                ROOT_DIR,
                self.config["MIL_MODEL"]["SNV_FEATERIZER"]["inhouse_freq_path"],
            ),
            logger=self.logger,
        )
        self.cnv_featurizer = CNVFeaturizer("wes")

        return

    def _check_same_artifact_uuid(self) -> bool:
        """저장된 메타데이터의 아티펙트()의 UUID을 읽어와 config에 저장된 UUID와 일치하는지 확인"""

        self.logger.info("Checking artifact UUID")
        meta_path = os.path.join(CHECKPOINT_DIR, "MLmodel")
        if not os.path.exists(meta_path):
            self.logger.info("MLmodel(%s) not exist" % meta_path)
            return False

        metadata = OmegaConf.load(meta_path)
        saved_uuid = metadata["model_uuid"]
        config_uuid = self.config["MIL_MODEL"]["UUID"]

        if saved_uuid == config_uuid:
            self.logger.info("Indentified same artifact UUID")
            return True

        self.logger.info("Indentified different artifact UUID")
        return False

    def download_artifact(self, model_config: Dict[str, Any]) -> None:
        """MLflow의 TrackingURI를 설정하고, 모델 설정 및 아티팩트 경로를 사용하여 메타데이터와
        체크포인트 및 스케일러 아티팩트를 다운로드

        Args:
            model_config (Dict[str, Any]): ARTIFACT_ROOT 및 각 artifact 이름이 기록된 dict

        Example:
            >>> from ASC3.mil_model.model import MILPredictor
            >>> mil_config = {"ARTIFACT_ROOT":..., }
            >>> MILPredictor.download_artifact(mil_config)
        """
        mlflow.set_tracking_uri(TRACKING_URI)
        log_if_exist = (
            lambda msg: self.logger.info(msg) if hasattr(self, "logger") else print(msg)
        )
        artifact_root = model_config["ARTIFACT_ROOT"]

        scaler_tracking_path = os.path.join(artifact_root, model_config["SCALER"])
        scaler_local_path = os.path.join(CHECKPOINT_DIR, model_config["SCALER"])
        if os.path.exists(scaler_local_path):
            os.remove(scaler_local_path)
        log_if_exist("Download scaler from mlflow tracking URI")
        path = mlflow.artifacts.download_artifacts(
            scaler_tracking_path, dst_path=CHECKPOINT_DIR
        )
        log_if_exist("Download complete: %s" % path)

        model_tracking_path = os.path.join(
            artifact_root, "data", model_config["CHECKPOINT"]
        )
        model_local_path = os.path.join(CHECKPOINT_DIR, model_config["CHECKPOINT"])
        if os.path.exists(model_local_path):
            os.remove(model_local_path)
        log_if_exist("Download checkpoint from mlflow tracking URI")
        path = mlflow.artifacts.download_artifacts(
            model_tracking_path, dst_path=CHECKPOINT_DIR
        )
        log_if_exist("Download complete: %s" % path)

        meta_tracking_path = os.path.join(artifact_root, model_config["METADATA"])
        meta_local_path = os.path.join(CHECKPOINT_DIR, model_config["METADATA"])
        if os.path.exists(meta_local_path):
            os.remove(meta_local_path)
        log_if_exist(
            "Download MLmodel from mlflow tracking URI(%s)" % meta_tracking_path
        )
        path = mlflow.artifacts.download_artifacts(
            meta_tracking_path, dst_path=CHECKPOINT_DIR
        )
        log_if_exist("Download complete: %s" % path)

        return

    def _set_model(self) -> None:
        """모델로딩"""

        mil_config = self.config["MIL_MODEL"]
        local_checkpoint_path = os.path.join(CHECKPOINT_DIR, mil_config["CHECKPOINT"])
        local_scaler_path = os.path.join(CHECKPOINT_DIR, mil_config["SCALER"])
        # local_calibrator_path = os.path.join(CHECKPOINT_DIR, mil_config["CALIBRATOR"])

        self.trials += 1
        if self.trials == 3:
            raise FileNotFoundError("checkpoint and scaler not found")

        if not self._check_same_artifact_uuid():
            try:
                self.download_artifact(mil_config)
                self._set_model()
            except mlflow.MlflowException:
                time.sleep(10)
                self._set_model()

        if os.path.exists(local_checkpoint_path) and os.path.exists(local_scaler_path):
            self.model: MultimodalAttentionMIL = torch.load(
                os.path.join(CHECKPOINT_DIR, mil_config["CHECKPOINT"]),
                map_location=self.device,
            )
            self.scalers = torch.load(
                os.path.join(CHECKPOINT_DIR, mil_config["SCALER"]),
                map_location=self.device,
            )["scaler"]
            # TODO load calibration model
            # self.calibration_model = ...
            self.logger.info("Set models and scaler as attribute")
            return

        else:
            self.logger.info(
                (
                    "local_checkpoint_path(%s) or " % local_checkpoint_path
                    + "local_scaler_path(%s) " % local_scaler_path
                    # + "local_calibrator_path(%s) " % local_calibrator_path
                    + "not found"
                )
            )
            self.download_artifact(mil_config)
            self._set_model()

        return

    def build_data_from_file(self, sample_id: str) -> PatientData:
        """파일로부터 환자 데이터(PatientData)를 생성

        Args:
            sample_id (str): 환자의 샘플 ID

        Returns:
            PatientData: 환자 데이터 객체
        """

        if not (hasattr(self, "snv_featurizer") and hasattr(self, "cnv_featurizer")):
            self._set_featurizer()

        self.logger.info("Retrieving HPO from dynamodb: sample_id %s" % sample_id)
        sample_hpo = [
            hpo for hpo in self.dynamodb_client.get_hpo(sample_id) if hpo != "-"
        ]

        self.logger.info("Build SNVData: sample_id")
        snv_data = self.snv_featurizer.build_data(
            sample_id,
            all_features=True,
        )

        cnv_data = self.cnv_featurizer.build_data(
            sample_id, list(), patient_hpos=sample_hpo
        )

        return PatientData(
            sample_id, bag_label=False, snv_data=snv_data, cnv_data=cnv_data
        )

    def make_snv_data(
        self, snv_query_data: Dict[str, Dict[str, SNVFeature]], inhouse_total_ac: int
    ) -> SNVData:
        """JSON 형식의 query로부터 SNVData을 생성

        Args:
            snv_query_data (Dict[str, SNVFeature]): SNV 데이터
            inhouse_total_ac (int): 3B내부의 전체 Allele count 수

        Returns:
            SNVData: snv_data
        """

        self.logger.info("Make SNVData from snv_qeury data")
        variants = list()
        vectors = list()
        for gene_disease, snv_feature in snv_query_data.items():
            self.logger.debug("gene_disease id(%s) process" % gene_disease)
            gene_id, disease_id = gene_disease.split("-", maxsplit=1)

            for cpra, features in snv_feature.items():
                vector = features.to_vector(inhouse_total_ac)
                self.logger.debug(
                    "CPRA(%s) with feature (%s)"
                    % (cpra, ",".join(map(lambda x: str(x), vector.tolist())))
                )

                vectors.append(vector)
                variants.append(
                    Variant(
                        cpra, acmg_rules=list(), gene_id=gene_id, disease_id=disease_id
                    )
                )

        return SNVData(
            x=np.vstack(vectors), variants=variants, header=self.feature_name
        )

    def make_cnv_data(self, cnv_query_data: Dict[str, CNVFeature]) -> CNVData:
        """클라이언트가 쿼리한 CNV 데이터를 사용하여 CNVData 객체를 생성

        Args:
            cnv_query_data (Dict[str, CNVFeature]): CNV 영역별 feature

        Returns:
            CNVData: 생성된 CNVData 객체.

        """
        variant = list()
        vectors = list()
        for region, cnv_feature in cnv_query_data.items():
            vectors.append(list(cnv_feature.dict().values()))
            variant.append(Variant(region, acmg_rules=list()))

        return CNVData(x=np.array(vectors), variants=variant)

    def convert_query_to_patient_data(self, mil_request: MILRequest) -> PatientData:
        """클라이언트 요청(MIL_request)로부터 예측할 PatientData 객체를 생성

        Args:
            mil_request (MILRequest): 클라이언트 요청

        Returns:
            PatientData: 예측할 PatientData
        """
        snv_data: SNVData = self.make_snv_data(
            mil_request.snv, inhouse_total_ac=mil_request.inhouse_total_ac
        )
        cnv_data: CNVData = self.make_cnv_data(mil_request.cnv)

        patient_data = PatientData(
            sample_id=mil_request.sample_id,
            bag_label=False,
            snv_data=snv_data,
            cnv_data=cnv_data,
        )

        return patient_data

    def truncate_prob(
        self, prob: float, t1: float = 0.001, t2: float = 0.0001
    ) -> float:
        """주어진 임계값 이하인 확률 값을 잘라내는 함수

        이 함수는 확률 값을 입력으로 받아, 지정된 임계값 `t1`보다 크거나 같은지 확인.
        입력 확률 값이 `t1`보다 크거나 같으면, 입력 확률 값을 그대로 반환. 만약 입력 확률 값이
        `t2`보다 크거나 같고 `t1`보다 작으면, 함수는 `t1`을 반환. 입력 확률 값이 `t2`보다 작으면,
        0을 반환.

        Parameters:
            prob (float): 잘라내려고 하는 입력 확률 값.
            t1 (float, optional): 임계값 1로, 기본값은 0.001입니다.
            t2 (float, optional): 임계값 2로, 기본값은 0.0001입니다.

        Retruns:
            float: 잘라낸 확률 값.

        Example:
            >>> truncate_prob(0.00005)
            0.0
            >>> truncate_prob(0.0005)
            0.001
            >>> truncate_prob(0.002)
            0.002
        """
        if prob >= t1:
            return prob

        elif prob >= t2:
            return 0.001

        else:
            return 0

    def post_process(
        self, instance_prob: np.ndarray, patient_data: PatientData
    ) -> Dict[str, Dict[str, float]]:
        """추론 결과를 후처리하여 변이와 점수로 이루어진 딕셔너리를 반환

        Args:
            instance_prob (np.ndarray): 인스턴스 확률값 텐서
            patient_data (PatientData): 환자 데이터 객체

        Returns:
            Dict[str, float]: 변이와 점수를 포함한 딕셔너리
        """

        snv_res = defaultdict(dict)
        n_snv = len(patient_data.snv_data.variants)
        for prob, variant in zip(instance_prob, patient_data.snv_data.variants):
            key = variant.gene_id + "-" + variant.disease_id
            snv_res[key][variant.cpra] = self.truncate_prob(prob.item())

        for key, value in snv_res.items():
            snv_res[key] = dict(sorted(value.items(), key=lambda x: x[1], reverse=True))

        snv_res = dict(
            sorted(snv_res.items(), key=lambda x: sum(x[1].values()), reverse=True)
        )

        cnv_res = dict()
        for prob, variant in zip(instance_prob[n_snv:], patient_data.cnv_data.variants):
            cnv_res[variant.cpra] = self.truncate_prob(prob.item())

        return {
            "snv": snv_res,
            "cnv": cnv_res,
        }

    def predict(self, patient_data: PatientData) -> Tuple[float, dict]:
        """
        환자 데이터를 기반으로 변이 예측을 수행하고 결과를 반환

        Args:
            patient_data (PatientData): 환자 데이터 객체

        Returns:
            Tuple[bool, dict]: 변이 예측 결과와 변이별 점수가 포함된 튜플
        """
        is_empty_cnv = False
        if len(patient_data.cnv_data.x) == 0:
            is_empty_cnv = True
            patient_data.cnv_data.x = np.zeros((1, 3), dtype=np.float32)

        dataset = ExSCNVDataset(
            PatientDataSet([patient_data]),
            base_features=self.config["MIL_MODEL"]["BASE_FEATURE"],
            additional_features=(
                self.config["MIL_MODEL"]["ADDITIONAL_FEATURES"]
                + self.config["MIL_MODEL"]["RULES"]
            ),
            scalers=self.scalers,
            device=self.device,
        )

        with torch.no_grad():
            (snv_x, cnv_x), _, _ = dataset[0]
            bag_logit, instance_logit = self.model((snv_x, cnv_x))
            bag_logit = bag_logit.cpu()
            instance_logit = instance_logit.cpu()

        bag_prob = torch.sigmoid(bag_logit).item()
        instance_prob = torch.sigmoid(instance_logit).numpy()

        # TODO
        # instance_prob = self.calibration_model.predict_proba(instance_prob)[:, 1]
        # instance_prob[instance_prob < 0.001 & instance_prob > 0.0001)] = 0.001

        variant2score = self.post_process(instance_prob, patient_data)

        if is_empty_cnv:
            variant2score["cnv"] = dict()

        return bag_prob, variant2score


class EnsembleMILPredictor(MILPredictor):
    """SNV에 대해서는 앙상블추론하는 추론용객체"""

    def __init__(
        self, config: dict, device: str = "cpu", logger: Logger = Logger(__name__)
    ) -> None:
        self.config = config
        self.logger = logger
        self.device = device
        self.trials = 0
        self._set_model()
        self._set_tree_model()
        self.feature_name = (
            self.config["MIL_MODEL"]["BASE_FEATURE"]
            + self.config["MIL_MODEL"]["ADDITIONAL_FEATURES"]
            + self.config["MIL_MODEL"]["RULES"]
        )

    def _set_tree_model(self):
        self.logger.info(
            "Load random forest from: %s" % self.config["MODEL"]["ARTIFACT_ROOT"]
        )
        if self.trials >= 3:
            raise mlflow.exceptions.MlflowException(
                "Random Forest download fail from MODEL.ARTIFACT_ROOT"
            )

        try:
            self.tree_model: RandomForestClassifier = mlflow.sklearn.load_model(
                os.path.join(CHECKPOINT_DIR, "rf_model")
            )

        except (mlflow.exceptions.MlflowException, OSError) as e:
            mlflow.set_tracking_uri(TRACKING_URI)
            mlflow.artifacts.download_artifacts(
                self.config["MODEL"]["ARTIFACT_ROOT"], dst_path=CHECKPOINT_DIR
            )
            self._set_tree_model()

        return

    def predict(self, patient_data: PatientData) -> Tuple[bool, dict]:
        """
        환자 데이터를 기반으로 변이 예측을 수행하고 결과를 반환

        Args:
            patient_data (PatientData): 환자 데이터 객체

        Returns:
            Tuple[bool, dict]: 변이 예측 결과와 변이별 점수가 포함된 튜플
        """
        is_empty_cnv = False
        if len(patient_data.cnv_data.x) == 0:
            is_empty_cnv = True
            patient_data.cnv_data.x = np.zeros((1, 3), dtype=np.float32)

        dataset = ExSCNVDataset(
            PatientDataSet([patient_data]),
            base_features=self.config["MIL_MODEL"]["BASE_FEATURE"],
            additional_features=(
                self.config["MIL_MODEL"]["ADDITIONAL_FEATURES"]
                + self.config["MIL_MODEL"]["RULES"]
            ),
            scalers=self.scalers,
            device=self.device,
        )

        with torch.no_grad():
            (snv_x, cnv_x), _, _ = dataset[0]
            bag_logit, instance_logit = self.model((snv_x, cnv_x))
            bag_logit = bag_logit.cpu()
            instance_logit = instance_logit.cpu()

        bag_prob = torch.sigmoid(bag_logit).item()
        instance_prob = torch.sigmoid(instance_logit).numpy()

        mil_snv_prob = instance_prob[: len(snv_x)]
        tree_snv_prob = self.tree_model.predict_proba(patient_data.snv_data.x[:, :6])[
            :, -1
        ].ravel()
        ensemble_snv_prob = (2 / 3 * tree_snv_prob) + (1 / 3 * mil_snv_prob)
        instance_prob[: len(snv_x)] = ensemble_snv_prob

        # TODO
        # instance_prob = self.calibration_model.predict_proba(instance_prob)[:, 1]
        # instance_prob[instance_prob < 0.001 & instance_prob > 0.0001)] = 0.001

        variant2score = self.post_process(instance_prob, patient_data)

        if is_empty_cnv:
            variant2score["cnv"] = dict()

        return bag_prob, variant2score
