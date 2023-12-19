import os
import sys
import math
import gzip
import json
import warnings

from pathlib import Path
from logging import Logger
from collections import Counter, OrderedDict
from typing import List, Optional, Tuple, Union, Dict, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder


CORE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CORE_DIR)
sys.path.append(ROOT_DIR)

from core.data_model import SNVData, Variant

warnings.filterwarnings("ignore")


def get_vaf(ad: str) -> float:
    """AD값을 VAF값으로 변환

    Args:
        ad: allele depth

    Return:
        vaf (float): variant allele fraction (=variant AD / DP)
    """
    try:
        ref, *alt = map(int, str(ad).split(","))
    except:
        return 0

    n_alt = sum(alt)
    n_total = n_alt + ref

    return n_alt / (n_total) if n_total != 0 else 0


def convert_rule_strength_to_vec(rule_strength: dict) -> List[int]:
    rules = [
        "PVS1",
        "PS1",
        "PS2",
        "PS3",
        "PS4",
        "PM1",
        "PM2",
        "PM3",
        "PM4",
        "PM5",
        "PM6",
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
    ]

    strength_to_int = {
        "VS": 0,
        "S": 1,
        "M": 2,
        "P": 3,
        "A": 4,
        "-": 5,
    }

    return [strength_to_int[rule_strength.get(rule, "-")] for rule in rules]


def parse_acmg(assigned_rule: str, delimiter: str = "||") -> dict:
    """구분문자(delimiter)로 연결된 ACMG_RULE을 파싱하여, RULE별 Strength로 변환

    Args:
        acmg_field (dict): ACMG

    Example:
        >>> parse_acmg("PVS1_VS|PS1_S|BP7_P")
        {
            "PVS1": "VS",
            "PS1": "S",
            ...
        }
    """
    if not isinstance(assigned_rule, str):
        return dict()

    if not assigned_rule:
        return dict()

    acmg_splited: list = assigned_rule.split(delimiter)

    return {rule.split("_")[0]: rule.split("_")[1] for rule in acmg_splited}


class VariantFilter:
    def __init__(self) -> None:
        """
        Example:
            >>> variant_filter = VariantFilter(...)
            >>> filtered_dataframe:pd.DataFrame = variant_filter.filter_variant(dataframe)

        """

    def _check_known_variant(self, row: pd.Series) -> bool:
        if "BP6" in row["ACMG_RULE"] and "BP6_-" not in row["ACMG_RULE"]:
            return False

        if "flag_variant:title" in row.index:
            flag_titles = row["flag_variant:title"].split("||")
            if "het_benign" in flag_titles:
                return False
            elif "homo/hemi_benign" in flag_titles and row["vcf_info_GT"] == "1/1":
                return False

        if row["clinvar_variant:pathogenicity"] != "-":
            gene_symbols = row["clinvar_variant:variant:gene:symbol"].split("||")
            for idx, word in enumerate(
                row["clinvar_variant:pathogenicity"].split("||")
            ):
                if (
                    word
                    in [
                        "Likely pathogenic",
                        "Pathogenic",
                        "Pathogenic/Likely pathogenic",
                    ]
                    and gene_symbols[idx] == row["officialSymbol"]
                ):
                    return True

        for rule in row["additional_variant:rule"].split("||"):
            if rule.startswith("P"):
                return True

        if row["hgmd_variant:pmid"] != "-":
            hgmd_symbols = row["hgmd_variant:gene_symbol"].split("||")
            hgmd_tags = row["hgmd_variant:tag"].split("||")
            for hgmd_symbol, hgmd_tag in zip(hgmd_symbols, hgmd_tags):
                if hgmd_symbol == row["officialSymbol"] and hgmd_tag in ["DM"]:
                    return True

        return False

    def check_known_variant(self, row: pd.Series) -> bool:
        try:
            return self._check_known_variant(row)
        except:
            return False

    def filter_variant(self, df: pd.DataFrame) -> pd.DataFrame:
        # replace consequence
        cond0 = (
            (df["HGVSp"].str.endswith("Ter"))
            & (~df["HGVSp"].str.contains("fs"))
            & (df["Consequence"].str.contains("frameshift"))
        )
        df.loc[cond0]["Consequence"] = df.loc[cond0]["Consequence"].replace(
            {"frameshift_variant": "stop_gained"}
        )

        cond1 = (
            (df["phenoId"] == "-")
            | (
                df["Feature"].str.startswith("ENS")
                & ~df["#Uploaded_variation"].str.startswith("MT-")
            )
            | (df["phenoId"].str.startswith("ORPHA"))
        )

        cond2 = (
            df["#Uploaded_variation"]
            .str.split("-")
            .apply(lambda x: len(x[2]) >= 2 and len(x[3]) >= 2)
        )

        df = df.loc[~(cond1 | cond2)]

        is_common = df["common_inhouse_1in100"].astype(str) == "True"
        cond3 = (
            (df["ACMG_class"].isin(["Benign", "Likely benign"]))
            | (df["ACMG_bayesian"].astype(float) <= 0.1)
            | (df["ACMG_RULE"].str.contains("BA1"))
            | is_common
        )

        cond4 = (
            (
                (df["exon:region_type"] != "cds")
                | (df["Consequence"].str.contains("synonymous"))
            )
            & ~(
                df["Consequence"].str.contains("splice_acceptor_variant")
                | df["Consequence"].str.contains("splice_donor_variant")
                | df["exon:distance"].isin({"1", "2"})
            )
            & (
                df["SPLICEAI-VALUE"].apply(
                    lambda x: float(x) < 0.1 if x != "-" else True
                )
            )
        )

        cond5 = df["#Uploaded_variation"].str.startswith("MT") & df[
            "Consequence"
        ].str.contains("stream")

        is_pathogenic = df["ACMG_class"].str.contains("athogenic")
        # TODO: include
        known_var = df.apply(lambda x: self.check_known_variant(x), axis=1)

        to_be_remain = is_pathogenic | known_var
        to_be_filtered = cond3 | cond4 | cond5

        after_exclude_df = df.loc[to_be_remain | ~to_be_filtered]

        return after_exclude_df


def read_clinvar_variants(path: str) -> dict:
    """Clinvar에 보고된 변이중 SVC(sumbission)기준으로 benign이 몇 건이 있었는지
    파싱함

    Args:
        path (str): clinvar path

    Returns:
        (dict): 변이별 benign 레포트수
    """
    res = dict()

    df = pd.read_csv(path, sep="\t", compression="gzip")
    for idx, row in df.iterrows():
        cpra = row["#normalized_variant:GRCh37:CPRA"]
        pathogenicities = row["scv:pathogenicity"].split(";;")

        n_benign = sum(
            ["benign" in pathogenicity.lower() for pathogenicity in pathogenicities]
        )

        res[cpra] = n_benign

    return res


class SNVFeaturizer:
    essential_cols = OrderedDict(
        [
            ("#Uploaded_variation", str),
            ("SYMBOL", str),
            ("symptom_similarity", str),
            ("vcf_info_QUAL", float),
            ("vcf_info_AD", str),
            ("inheritance", str),
            ("vcf_info_GT", str),
            ("phenoId", str),
            ("ACMG_RULE", str),
            ("ACMG_bayesian", float),
        ]
    )
    base_features = [
        "ACMG_bayesian",
        "symptom_similarity",
        "vcf_info_QUAL",
        "inhouse_freq",
        "vaf",
        "is_incomplete_zygosity",
    ]

    snv_root_path = (
        ""
    )
    snv_38_root_path = (
        ""
    )

    with open(CORE_DIR + "/snv_fields.json", "r") as file:
        evidence_fields = json.load(file)

    all_fields: List[str] = (
        evidence_fields["numerical_fields"]
        + evidence_fields["acmg_rules"]
        + list(evidence_fields["categorical_fields"].keys())
    )

    def __init__(
        self,
        sequencing: Literal["wes", "wgs"] = "wes",
        logger: Optional[Logger] = Logger(__file__),
        inhouse_freq_path: Optional[Union[str, Path]] = None,
    ):
        """
        args :
            sequencing (Literal["wes", "wgs"]): 시퀀싱 유형
                "wes" (Whole Exome Sequencing) 또는 "wgs" (Whole Genome Sequencing) 중 하나
            snv_root_path (str, optional): evidence result path
            logger (Logger, optional): 로깅을 위한 Logger 객체를 지정
            inhouse_freq_path (str, optional): WES/WGS에 따른 inhouse_freq file path

        Example:
            >>> snv_featurizer = SNVFeaturizer(
                    snv_root_path=".../result",
                    inhouse_freq_path="...B.ADZQ.variant.count.table.txt
                )
            >>> snv_featurizer.build_data(sample_id="EPD21-ESPN")

        """
        self.sequencing = sequencing
        self.variant_filter = VariantFilter()
        self.logger = logger

        if inhouse_freq_path:
            self.inhouse_freq_path = Path(inhouse_freq_path).resolve()
            self.inhouse_freq_dict = self.load_inhouse_freq_dict(self.inhouse_freq_path)

    def load_inhouse_freq_dict(self, file_path: Union[str, Path]) -> None:
        """사내용 Inhouse frequency 데이터를 불러옴"""
        self.logger.info(
            f"Load inhouse frequency: {file_path}"
        ) if self.logger else None

        with open(file_path, "r") as fh:
            for idx, line in enumerate(fh):
                row = line.strip().split("\t")
                if idx == 1:
                    _, _, male_n, female_n = row

                if line.startswith("# variant"):
                    columns = line.lstrip("# ").strip().split("\t")
                    break

        total_n = int(male_n) + int(female_n)
        data = pd.read_csv(file_path, skiprows=(range(0, 5)), sep="\t", names=columns)

        trim_col = "female.homo.AC" if self.sequencing == "wes" else "female.homo"
        data = data.iloc[:, : data.columns.tolist().index(trim_col) + 1]

        inhouse_freq = data.set_index("variant").astype("int16").sum(axis=1) / total_n
        return inhouse_freq.to_dict()

    def read_tsv_file(self, file_path: Path) -> pd.DataFrame:
        file_is_missing = not (file_path.is_file())
        if file_is_missing:
            return None

        skiprows = 0
        with gzip.open(file_path, "rt") as textio:
            while True:
                line = textio.readline()
                if not (line.startswith("##")):
                    columns = line.strip("\n").split("\t")
                    break
                skiprows += 1

        patient_snv_df = pd.read_csv(
            file_path,
            compression="gzip",
            sep="\t",
            low_memory=False,
            skiprows=skiprows,
        )

        return patient_snv_df

    def get_incomplete_zygosity(
        self, symbol_cnt: int, inheritances: Union[str, float], gt: str
    ) -> bool:
        """환자의 성별에 맞춰 incomplete zygoisty을 call-by-reference로 할당

        Note:
            Incomplete zygosity 기준은 GEBRA에 올라온 candidate variant을 기준으로함
            (임상팀으로부터 확인함. 23.02.13. Austin)
            - Good zygosity 정의 (서샘 컨펌)
            1. Homo: AR유전자에서 변이가 GT가 1/1인 경우
            2. Hetero: AR유전자에서 변이가 적어도 2개 이상 있는 경우
            (Cis/Trans인건 일단 구별 안함)

            - incomplete zygosity 정의
            1. Homo : AR 유전자에서 변이가 최소 2개 이상인경우는
            2. Hetro : AR 유전자에서 GT가 1/0인 변이가 1개밖에 없는 경우
            (Cis/Trans인건 일단 구별 안함)

        Args:
            symbol_cnt (int): 환자 VCF 내의 해당 gene symbol의 개수.
            inheritance (Union[str, float]): inheritance pattern.
            gt (str): genotype.

        Return:
            bool: incomplete zygosity 여부, True if incomplete zygosity

        Example:
            >>> cls.get_incomplete_zygosity(1, "Autosomal recessive||Somatic mutation", "0/1")
            True

        """

        if symbol_cnt >= 2:
            return False

        inheritances_is_nan = (isinstance(inheritances, float)) and (
            math.isnan(inheritances)
        )
        if (inheritances == "-") or inheritances_is_nan:
            return False

        is_one_copy_affected = gt.count("1") == 1
        inheritances = inheritances.split("||")

        return (
            "Autosomal dominant" not in inheritances
            and "Autosomal recessive" in inheritances
            and is_one_copy_affected
        )

    def parse_acmg_pipe(
        self, assigned_rule: pd.Series, delimiter: str = "||"
    ) -> pd.Series:
        """구분문자(delimiter)로 연결된 ACMG_RULE을 파싱하여, RULE별 Strength로 변환

        Args:
            acmg_field (dict): _description_

        Example:
            >>> self.parse_acmg("PVS1_VS||PS1_S||BP7_P")
            {
                "PVS1": "VS",
                "PS1": "S",
                ...
            }
        """

        return assigned_rule.apply(lambda x: parse_acmg(str(x), delimiter=delimiter))

    def convert_rule_strength_to_vec_pipe(self, series: pd.Series) -> pd.Series:
        return series.apply(lambda x: convert_rule_strength_to_vec(x))

    def count_pathogenicity(self, x):
        try:
            terms = x.split(",")
        except:
            return 0, 0
        n_benign = 0
        n_pathogenic = 0
        for term in terms:
            term: str = term.lower()
            if "benign" in term:
                n_benign += 1
            elif "pathogenic" in term:
                n_pathogenic += 1
        return n_benign, n_pathogenic

    def featurize_all(self, df: pd.DataFrame) -> pd.DataFrame:
        snv_df = df.copy(deep=True)

        vectorized_rules: pd.Series = pd.DataFrame(
            snv_df["ACMG_RULE"]
            .pipe(self.parse_acmg_pipe)
            .pipe(self.convert_rule_strength_to_vec_pipe)
            .tolist(),
            columns=self.evidence_fields["acmg_rules"],
        )

        for consequence in self.evidence_fields["consequences"]:
            try:
                snv_df[consequence] = snv_df["Consequence"].apply(
                    lambda x: consequence in x.split(",")
                )
            except:
                snv_df[consequence] = "False"

        snv_df["vaf"] = snv_df["vcf_info_AD"].apply(lambda x: get_vaf(x))
        n_benign, n_pathogenic = zip(
            *snv_df["clinvar_variant:scv:pathogenicity"].apply(
                lambda x: self.count_pathogenicity(x)
            )
        )
        snv_df["clinvar_variant:scv:pathogenicity_n_p"] = n_pathogenic
        snv_df["clinvar_variant:scv:pathogenicity_n_b"] = n_benign

        splited_stars = snv_df[f"clinvar_variant:star"].apply(lambda x: x.split("||"))
        for n_stars in range(5):
            snv_df[f"clinvar_variant:star_{n_stars}"] = splited_stars.apply(
                lambda x: x.count(str(n_stars))
            )

        for feature_name, categories in self.evidence_fields[
            "categorical_fields"
        ].items():
            if feature_name in self.evidence_fields["acmg_rules"]:
                continue

            l_enc = LabelEncoder()
            l_enc.classes_ = np.array(list(categories))
            snv_df[feature_name] = snv_df[feature_name].fillna("-")

            try:
                snv_df[feature_name] = l_enc.transform(snv_df[feature_name].values)
            except TypeError:
                if snv_df[feature_name].dtype == bool:
                    continue
                else:
                    raise ValueError

        for col_name in self.evidence_fields["numerical_fields"]:
            snv_df[col_name] = pd.to_numeric(snv_df[col_name], errors="coerce")

        concat_df = pd.concat([snv_df.reset_index(drop=True), vectorized_rules], axis=1)
        return concat_df[self.all_fields].astype(float).fillna(value=-1)

    def annotated_label(
        self, df: pd.DataFrame, causal_variant: List[Tuple]
    ) -> pd.DataFrame:
        """
        주어진 데이터프레임에 주어진 원인 변이를 기반으로 레이블을 표기

        Args:
            df (pd.DataFrame): 레이블을 부착할 데이터프레임입니다.
            causal_variant(List[Tuple]): 인과적 변이를 나타내는 정보입니다.

        Returns:
            pd.DataFrame: 데이터프레임
        """

        df["label"] = False
        df.loc[
            df[["#Uploaded_variation", "phenoId"]]
            .apply(tuple, axis=1)
            .isin(causal_variant),
            "label",
        ] = True

        return df

    def annotated_base_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        데이터프레임에 기본 특징화작업을 진행

        Args:
            df (pd.DataFrame): 기능을 부착할 데이터프레임입니다.

        Returns:
            pd.DataFrame: 기능이 부착된 데이터프레임을 반환합니다.
        """

        # inhouse frequency
        df["inhouse_freq"] = df["#Uploaded_variation"].apply(
            lambda cpra: self.inhouse_freq_dict.get(cpra, 0)
        )

        # vaf
        df["vaf"] = [get_vaf(ad) for ad in df["vcf_info_AD"].values]

        # incomplete zygosity
        symbol_count = Counter(df["SYMBOL"].values)
        df["SYMBOL_CNT"] = df["SYMBOL"].apply(lambda x: symbol_count[x])
        df["is_incomplete_zygosity"] = df[
            ["SYMBOL_CNT", "inheritance", "vcf_info_GT"]
        ].apply(lambda x: self.get_incomplete_zygosity(*x), axis=1)

        # symptom similarity
        df["symptom_similarity"].replace("-", "0", inplace=True)
        df["symptom_similarity"] = df["symptom_similarity"].astype(np.float32)

        return df

    def reduce_transcript(self, df: pd.DataFrame) -> pd.DataFrame:
        """중복된 Transcripts 을 삭제

        Args:
            df (pd.DataFrame): 환자 변이테이블

        Returns:
            pd.DataFrame: 중복된 transcript가 삭제된 환자 변이 테이블

        """
        # Uploaded_variation, phenoId 기준으로 unique, ACMG_bayesian 높은 것만
        df.sort_values(
            ["ACMG_bayesian", "phenoId", "#Uploaded_variation"],
            ascending=False,
            inplace=True,
        )
        df.drop_duplicates(
            ["phenoId", "#Uploaded_variation"], keep="first", inplace=True
        )

        return df

    def _calculate_features(self, file_path: Union[Path, str], *args) -> SNVData:
        causal_variant, all_features, additional_func = args
        patient_snv_df: pd.DataFrame = self.read_tsv_file(file_path)

        default_return = SNVData(causal_variant=causal_variant)

        # 파일이 없거나 필수키가 누락일 경우
        if patient_snv_df is None or any(
            col not in patient_snv_df.columns for col in self.essential_cols.keys()
        ):
            self.logger.info("File (%s) not found" % str(file_path))
            return default_return

        try:
            patient_snv_df: pd.DataFrame = self.variant_filter.filter_variant(
                patient_snv_df
            )
        except:
            self.logger.warning(f"In filtering variant, exception raised")
            return default_return

        patient_snv_df = (
            patient_snv_df.pipe(self.annotated_label, causal_variant=causal_variant)
            .pipe(self.reduce_transcript)
            .pipe(self.annotated_base_feature)
        )

        # 원인변이의 증상유사도 점수가 없는 경우
        if (
            patient_snv_df.loc[patient_snv_df["label"]]["symptom_similarity"] == "-"
        ).any():
            self.logger.warning(f"symptom_similarity of casual variant not found")
            return default_return

        feature_df = patient_snv_df[self.base_features]

        # Additional features
        for func in additional_func:
            feature_df = pd.concat(
                [feature_df, func(patient_snv_df)], ignore_index=True, axis=1
            )

        # Entire feautres
        entire_features = pd.DataFrame()
        if all_features:
            try:
                entire_features = self.featurize_all(patient_snv_df)
            except:
                self.logger.info(
                    "Failed to featurize_all from raw tsv data in %s" % str(file_path)
                )
                return default_return

        feature_df = pd.concat(
            [feature_df.reset_index(drop=True), entire_features], axis=1
        )

        return SNVData(
            causal_variant=causal_variant,
            x=feature_df.values,
            tsv_path=os.path.realpath(file_path),
            y=patient_snv_df["label"].values.astype(np.float32),
            header=list(feature_df.columns),
            variants=[
                Variant(
                    cpra=row["#Uploaded_variation"],
                    disease_id=row["phenoId"],
                    acmg_rules=[
                        rule
                        for rule in row["ACMG_RULE"].split("||")
                        if not rule.endswith("-")
                    ],
                    gene_id=row["geneId"],
                )
                for idx, row in patient_snv_df.iterrows()
            ],
        )

    def build_data(
        self,
        sample_id: str,
        causal_variant: List[Tuple[str, str]] = [("-", "-")],
        all_features: bool = False,
        additional_func: List[callable] = list(),
    ) -> SNVData:
        """

        Example:
            >>> def attach_gene_similary(df, sample_id) -> pd.Series:
                    client = DynamoDBClient("...")
                    hpos = {Phenotype(hpo) for hpo in client.get_hpo(sample_id)}
                    gene_similarity = calculator.get_gene_patient_symtom_similarity(
                        hpos, return_symbol=True
                    )

                return df["SYMBOL"].apply(lambda x: gene_similarity.get(x, 0)).rename("gene_similarity")

            >>> snv_data = featurizer.build_data(
                    sample_id="ETA22-SNRQ",
                    all_features=True,
                    additional_func=[partial(attach_gene_similary, sample_id="ETA22-SNRQ")],
                )

            >>> print(snv_data.x.shape)
            (209, 174)

        """
        prefix = sample_id.split("-")[1][:2]
        file_path = (
            Path(self.snv_38_root_path.format(prefix, sample_id, sample_id))
            if sample_id.startswith("G")
            else Path(self.snv_root_path.format(prefix, sample_id, sample_id))
        )
        return self._calculate_features(
            file_path, sample_id, causal_variant, all_features, additional_func
        )

    def build_data_from_tsv(
        self,
        sample_tsv_path: Path,
        causal_variant: List[Tuple[str, str]] = [("-", "-")],
        all_features: bool = False,
        additional_func: List[callable] = list(),
    ):
        """

        Example:
            >>> def attach_gene_similary(df, sample_id) -> pd.Series:
                    client = DynamoDBClient("...")
                    hpos = {Phenotype(hpo) for hpo in client.get_hpo(sample_id)}
                    gene_similarity = calculator.get_gene_patient_symtom_similarity(
                        hpos, return_symbol=True
                    )

                return df["SYMBOL"].apply(lambda x: gene_similarity.get(x, 0)).rename("gene_similarity")

            >>> snv_data = featurizer.build_data(
                    sample_tsv_path=".../ETA22-SNRQ.uploaded_data.tsv"
                    sample_id="ETA22-SNRQ",
                    all_features=True,
                    additional_func=[partial(attach_gene_similary, sample_id="ETA22-SNRQ")],
                )

            >>> print(snv_data.x.shape)
            (209, 174)

        """
        return self._calculate_features(
            sample_tsv_path, causal_variant, all_features, additional_func
        )


class SNVFeaturizerA4C(SNVFeaturizer):
    """닥터앤서 과제용 SNVFeaturizer"""

    input_feature_header = [
        "bayesian",
        "symptom_similarity",
        "vcf_info_QUAL",
        "inhouse_freq",
        "vaf",
        "is_incomplete_zygosity",
        "gnomad_ac",
        "clinvar_n_benign",
    ]

    def __init__(
        self,
        logger: Logger,
        sequencing: Literal["wes", "wgs"],
        inhouse_freq_path: str,
        snv_root_path: str = "./",
        include_rules=["PVS1", "PS1", "PS4", "PP3", "PP5", "BP4"],
    ):
        super().__init__(sequencing, snv_root_path, logger, inhouse_freq_path)
        self.include_rules = include_rules

    def annotated_symptom_similarity(
        self,
        input_dataframe: pd.DataFrame,
        disease_similarity: Dict[str, float],
    ) -> None:
        """입력 데이터프레임에 질병 유전자와의 유사도를 추가합니다.

        Args:
            input_dataframe (pd.DataFrame): 유전자 정보가 담긴 입력 데이터프레임
            disease_similarity (Dict[str, float]): 질병 유전자와의 유사도를 담은 딕셔너리

        Returns:
            None
        """

        input_dataframe["phenoId"] = input_dataframe["phenoId"].astype(str)
        ontology_pheno_id = input_dataframe["phenoId"].apply(
            lambda x: "OMIM:" + x if x.isdigit() else x
        )

        input_dataframe["symptom_similarity"] = ontology_pheno_id.apply(
            lambda x: disease_similarity.get(x, 0)
        )

        return

    def to_onehot(self, rules: str) -> np.ndarray:
        """Mini evidence RULE을 onehot vector로 표현

        Note:
            mini evidence에서 ACMG assigner가 예측한 결과가 없으면 디폴트 "-"

        Args:
            rules (str): mini evidence 결과의 RULE

        Returns:
            np.ndarray: onehot vecctor, shape=(n rules, )

        Example:
            >>> self.to_onehot("PVS1_VS")
            np.array([1, 0, 0, 0, 0, 0])
            >>> self.to_onehot("PVS1_VS||PP3_P")
            np.array([1, 0, 0, 1, 0, 0])
        """
        if rules == "-":
            return np.zeros(shape=(6,), dtype=np.float32)

        res = np.zeros(shape=(6,), dtype=np.float32)
        rules_wo_strength = {rule.split("_")[0] for rule in rules.split("||")}
        for idx, include_rule in enumerate(self.include_rules):
            if include_rule not in rules_wo_strength:
                continue

            res[idx] = 1

        return res

    def predict_full_acmg_bayesian_score(
        self, input_dataframe: pd.DataFrame, acmg_predictor: BaseEstimator
    ) -> np.ndarray:
        """mini-evidence 결과로 FULL ACMG Bayesian score을 예측

        Args:
            input_dataframe (pd.DataFrame): mini-evidence 결과가 저장된 데이터프레임
            acmg_predictor (BaseEstimator): FULL ACMG Bayesian score 예측기

        Returns:
            np.ndarray: FULL ACMG Bayesian score 예측 결과. shape=(N,)
        """

        rule_onehot = (
            input_dataframe["acmg"].apply(lambda x: self.to_onehot(x)).values
        )  # (N, )
        rule_onehot = np.stack(rule_onehot)  # (N, 6)
        base_acmg_score = (
            input_dataframe["bayesian"].values.astype(np.float32).reshape(-1, 1)
        )

        partial_acmg_features = np.concatenate([rule_onehot, base_acmg_score], axis=-1)

        return acmg_predictor.predict(partial_acmg_features).ravel()

    def set_clinvar_variants(self, path):
        self.clinvar_variants = read_clinvar_variants(path)
        return

    def set_gnomad_variant(self, path):
        self.gnomad_variants = pd.read_csv(path).set_index("cpra")["ac"].to_dict()
        return

    def annotate_features(
        self, input_dataframe: pd.DataFrame, acmg_predictor: BaseEstimator
    ) -> pd.DataFrame:
        num_before_drop = len(input_dataframe)

        mini_evi_indices = input_dataframe["preassigned_acmg"] == "-"

        input_dataframe.loc[~mini_evi_indices, "bayesian"] = input_dataframe.loc[
            ~mini_evi_indices, "preassigned_bayesian"
        ]

        input_dataframe = input_dataframe.sort_values(
            "bayesian", ascending=False
        ).drop_duplicates(["phenoId", "#Uploaded_variation"], keep="first")
        self.logger.debug(
            "{} out of {} variants are left after dropping duplicates.".format(
                len(input_dataframe), num_before_drop
            )
        )

        mini_evi_indices = input_dataframe["preassigned_acmg"] == "-"
        if mini_evi_indices.sum() != 0:
            input_dataframe.loc[
                mini_evi_indices, "bayesian"
            ] = self.predict_full_acmg_bayesian_score(
                input_dataframe.loc[mini_evi_indices], acmg_predictor
            )

        # inhouse freq
        input_dataframe["inhouse_freq"] = input_dataframe["#Uploaded_variation"].apply(
            lambda cpra: round(self.inhouse_freq_dict.get(cpra, 0), 7)
        )

        # vaf
        input_dataframe["vaf"] = [
            get_vaf(ad) for ad in input_dataframe["vcf_info_AD"].values
        ]

        # incomplete zygosity
        symbol_count = Counter(input_dataframe["SYMBOL"].values)
        input_dataframe["SYMBOL_CNT"] = input_dataframe["SYMBOL"].apply(
            lambda x: symbol_count[x]
        )
        input_dataframe["is_incomplete_zygosity"] = input_dataframe[
            ["SYMBOL_CNT", "inheritance", "vcf_info_GT"]
        ].apply(lambda x: self.get_incomplete_zygosity(*x), axis=1)

        input_dataframe["clinvar_n_benign"] = input_dataframe[
            "#Uploaded_variation"
        ].apply(lambda x: self.clinvar_variants.get(x, 0))
        input_dataframe["gnomad_ac"] = input_dataframe["#Uploaded_variation"].apply(
            lambda x: self.gnomad_variants.get(x, 0)
        )
        return input_dataframe

    def build_data_from_df(
        self,
        input_dataframe: pd.DataFrame,
    ) -> SNVData:
        missing_fields = list()
        for essential_col in self.input_feature_header + [
            "#Uploaded_variation",
            "acmg",
        ]:
            if essential_col in input_dataframe.columns:
                continue

            missing_fields.append(essential_col)

        if missing_fields:
            msg = (
                f"There're several missing fields({missing_fields}) in input dataframe."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        feature_x = input_dataframe[self.input_feature_header]

        variants = list()
        for _, row in input_dataframe.iterrows():
            pheno_id = str(row["phenoId"])
            prefix_pheno_id = "OMIM:" + pheno_id if pheno_id.isdigit() else pheno_id
            variants.append(
                Variant(
                    cpra=row["#Uploaded_variation"],
                    disease_id=prefix_pheno_id,
                    acmg_rules=[
                        rule
                        for rule in row["acmg"].split("||")
                        if not rule.endswith("-")
                    ],
                )
            )

        return SNVData(
            x=feature_x.values.astype(np.float32),
            header=self.input_feature_header,
            variants=variants,
        )
