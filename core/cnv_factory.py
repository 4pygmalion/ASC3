import gzip
import time
from logging import Logger
from pathlib import Path
from typing import Union, Dict, List, Set, Literal

import numpy as np
import pandas as pd

from SemanticSimilarity.calculator import cal_symtpom_similarity_from_lambda
from core.data_model import CNVData, Variant


class CNVFeaturizer:
    cnv_root_path = (
        ""
    )
    cnv_38_root_path = (
        ""
    )
    cnv_json_root_path = ""

    hpo2disease_path = Path("")
    header = ["ACMG_bayesian", "symptom_similarity", "num_genes"]

    def __init__(
        self,
        sequencing: Literal["wes", "wgs", "all"] = "wes",
        logger: Logger = Logger(__name__),
    ):
        self.logger = logger
        self.sequencing = sequencing
        self.hpo2disease_df = pd.read_csv(
            self.hpo2disease_path,
            compression="gzip",
            sep="\t",
            usecols=["#omimPhenoId", "geneSymbol"],
        )

    def read_tsv(self, file_path: Union[Path, str]) -> pd.DataFrame:
        """
        Note:
            2023-05-09 이후부터는 CNV 데이터를 /... 이하에 저장하고, json포맷대신에
            tsv파일을 저장함. 그리고 이 tsv파일을 S3 버켓에 올림.

            2023-11-17 이후부터는 /..., /...에 나누어 저장함.

            --TSV파일 포맷--
            ##Called: Conifer=14;Manta=0;3bCNV=8;Total=22
            ##Remained: Conifer=2;Manta=0;3bCNV=1;Total=3
            #chrom  start   end     cnv_type        caller  quality genotype        allele_depth...
            15      23606367        28623955        DEL     Conifer -       -       -       -   ...
            15      23606367        28623955        DEL     Conifer -       -       -       -   ...
            15      23606367        28623955        DEL     Conifer -       -       -       -   ...
            15      23606367        28623955        DEL     Conifer -       -       -       -   ...
            15      23606367        28623955        DEL     Conifer -       -       -       -   ...
        """
        skiprows = 0
        with gzip.open(file_path, "rt") as fh:
            while True:
                line = fh.readline()
                if not line.startswith("#chrom"):
                    skiprows += 1
                    continue

                break

        df: pd.DataFrame = pd.read_csv(
            file_path,
            sep="\t",
            compression="gzip",
            skiprows=skiprows,
        )

        return df

    def get_symptom_similarity(
        self, gene_symbols: Set[str], disease_similiarty: Dict[str, float]
    ) -> float:
        # patient's cnv associated disease
        gene_symbol_matched = self.hpo2disease_df["geneSymbol"].apply(
            lambda x: x in gene_symbols
        )
        matched_df: pd.DataFrame = self.hpo2disease_df[gene_symbol_matched]
        matched_df = matched_df[matched_df["#omimPhenoId"] != "-"]
        if len(matched_df) == 0:
            return 0
        matched_df["#omimPhenoId"] = matched_df["#omimPhenoId"].apply(
            lambda x: f"OMIM:{x}"
        )

        # query cnv disease in df
        matched_df["semantic_similarity"] = matched_df["#omimPhenoId"].apply(
            lambda omimid: disease_similiarty.get(omimid, 0)
        )
        # query 결과 내에 omimPhenoId 존재하지않는 경우 0으로 대치
        max_sim = matched_df["semantic_similarity"].astype(np.float32).max()
        del matched_df

        return max_sim

    def _calculate_features(self, file_path: Union[Path, str], *args) -> CNVData:
        causal_variant, patient_hpos = args
        self.logger.debug(
            "_calculate_features sample_id(%s), causal_variant(%s)"
            % (str(file_path), {",".join(causal_variant)})
        )
        default_return = CNVData(
            x=-np.ones((1, len(self.header))).astype(np.float32),
            y=np.zeros((1,)).astype(np.float32),
            causal_variant=causal_variant,
            header=self.header,
        )

        try:
            df = self.read_tsv(file_path)
        except:
            return default_return

        if df.empty:
            return default_return

        if "symptom_similarity" not in df.columns:
            disease_similiarty = dict()
            trials = 0
            while trials < 5:
                try:
                    disease_similiarty: dict = cal_symtpom_similarity_from_lambda(
                        patient_hpos
                    )
                except:
                    time.sleep(5)

                trials += 1

            if not disease_similiarty:
                self.logger.warning(
                    "Failed to calculate symptom similarity for file(%s)" % file_path
                )
                return default_return

            df["symptom_similarity"] = df["phenoId"].map(disease_similiarty)

        x = list()
        y = list()
        variants_names = list()
        varaints = df.apply(lambda x: f"{x['#chrom']}:{x['start']}-{x['end']}", axis=1)
        for variant in set(varaints):
            variant_rows = df.loc[varaints == variant]

            if len(variant_rows) == 0:
                continue

            not_missing_symbols = variant_rows["all_coding_symbols"][
                variant_rows["all_coding_symbols"] != "-"
            ]
            n_genes = not_missing_symbols.str.split("::").str.len().values

            similarities = [
                float(sim)
                for sim in variant_rows["symptom_similarity"]
                if sim != "-" and sim != "."
            ]
            rules = variant_rows["merged_acmg_rules"].tolist()[0]

            score = max(
                [
                    float(score)
                    for score in set(variant_rows["merged_acmg_sum"])
                    if score != "-"
                ]
            )
            max_n_gene = max(n_genes)
            symptom_similarity = max(similarities) if similarities else 0

            x.append([score, symptom_similarity, max_n_gene])
            y.append(False if variant not in causal_variant else True)
            variants_names.append(Variant(variant, acmg_rules=rules))

        # 원인변이는 있는데, 데이터셋내에 존재하지 않는 경우는 제외
        if causal_variant and sum(y) == 0:
            self.logger.warning("Causal CNV of sample(%s) not found." % str(file_path))
            return default_return

        return CNVData(
            causal_variant=causal_variant,
            x=np.array(x).astype(np.float32),
            y=np.array(y).astype(np.float32),
            variants=variants_names,
            header=self.header,
        )

    def build_data(
        self,
        sample_id: str,
        causal_variant: List[str] = list(),
        patient_hpos: List[str] = list(),
    ) -> CNVData:
        """CNV 데이터를 생성하여 반환

        Note:
            --features--
            1. 유전자수: 변이가 커버하는 유전자수는 대표변이와 동일한 변이가 커버하는 유전자수를 의미
            2. 증상유사도: 변이의 유전자의 여러 질환중, 증상유사도가 가장 높은 값
            3. ACMG score: AMCG 스코어의 최대값(동일한 변이면 모두 같은 듯)

        Args:
            sample_id (str): 샘플 ID 문자열.
            causal_variant (List[str], 선택적): 인과 유전자 변이 리스트. 기본값은 ["-"].

        Return:
            CNVData: CNV 데이터 객체.

        Examples:
            >>> featurizer = CNVFeaturizer(...)
            >>> featurizer.build_data("EPB23-LMKB", ["5:96244664-96271878"])
            CNVData(n_variants=3, causal_variant=["5:96244664-96271878"])
        """
        prefix = sample_id.split("-")[1][:2]
        file_path = (
            Path(self.cnv_38_root_path.format(prefix, sample_id, sample_id))
            if sample_id.startswith("G")
            else Path(self.cnv_root_path.format(prefix, sample_id, sample_id))
        )
        if not file_path.exists():
            return self.build_data_from_json(
                Path(self.cnv_json_root_path.format(prefix, sample_id, sample_id)),
                causal_variant,
                patient_hpos,
            )
        return self._calculate_features(file_path, causal_variant, patient_hpos)

    def build_data_from_tsv(
        self,
        sample_tsv_path: Union[str, Path],
        causal_variant: List[str] = list(),
        patient_hpos: List[str] = list(),
    ) -> CNVData:
        """CNV 데이터를 생성하여 반환

        Note:
            --features--
            1. 유전자수: 변이가 커버하는 유전자수는 대표변이와 동일한 변이가 커버하는 유전자수를 의미
            2. 증상유사도: 변이의 유전자의 여러 질환중, 증상유사도가 가장 높은 값
            3. ACMG score: AMCG 스코어의 최대값(동일한 변이면 모두 같은 듯)

        Args:
            sample_tsv_path: [sample_id].uploaded.tsv 파일의 경로
            sample_id (str): 샘플 ID 문자열.
            causal_variant (List[str], 선택적): 인과 유전자 변이 리스트. 기본값은 ["-"].

        Return:
            CNVData: CNV 데이터 객체.

        Examples:
            >>> featurizer = CNVFeaturizer(...)
            >>> featurizer.build_data(".../EPB23-LMKB.uploaded.tsv", ["5:96244664-96271878"])
            CNVData(n_variants=3, causal_variant=["5:96244664-96271878"])
        """
        return self._calculate_features(sample_tsv_path, causal_variant, patient_hpos)

    def build_data_from_json(
        self,
        cnv_json_path: str,
        causal_variant: List[str] = list(),
        patient_hpos: List[str] = list(),
    ) -> CNVData:
        try:
            patient_cnv_df = pd.read_json(cnv_json_path)
        except ValueError:
            self.logger.debug("File(%s) is invalid" % cnv_json_path)
            return CNVData(
                x=-np.ones((1, len(self.header))).astype(np.float32),
                y=np.zeros((1,)).astype(np.float32),
                causal_variant=causal_variant,
                header=self.header,
            )
        except FileNotFoundError:
            self.logger.debug(f"File(%s) not found" % cnv_json_path)
            return CNVData(
                x=-np.ones((1, len(self.header))).astype(np.float32),
                y=np.zeros((1,)).astype(np.float32),
                causal_variant=causal_variant,
                header=self.header,
            )
        patient_cnv_df["label"] = False
        if "pos" not in patient_cnv_df:
            return CNVData(
                x=-np.ones((1, len(self.header))).astype(np.float32),
                y=np.zeros((1,)).astype(np.float32),
                causal_variant=causal_variant,
                header=self.header,
            )
        patient_cnv_df.loc[patient_cnv_df["pos"].isin(causal_variant), "label"] = True
        disease_similiarty = dict()
        trials = 0
        while trials < 5:
            try:
                disease_similiarty: dict = cal_symtpom_similarity_from_lambda(
                    patient_hpos
                )
            except:
                time.sleep(5)
            trials += 1
        if not disease_similiarty:
            self.logger.warning(
                "Failed to calculate symptom similarity for %s" % cnv_json_path
            )
            return CNVData(
                x=-np.ones((1, len(self.header))).astype(np.float32),
                y=np.zeros((1,)).astype(np.float32),
                causal_variant=causal_variant,
                header=self.header,
            )
        patient_cnv_df["symptom_similarity"] = patient_cnv_df["genes"].apply(
            lambda x: self.get_symptom_similarity(x, disease_similiarty)
        )
        return CNVData(
            causal_variant=causal_variant,
            x=patient_cnv_df[
                ["score", "symptom_similarity", "num_genes"]
            ].values.astype(np.float32),
            y=patient_cnv_df["label"].values.astype(np.float32),
            header=self.header,
        )
