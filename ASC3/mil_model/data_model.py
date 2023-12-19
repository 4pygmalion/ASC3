import os
import sys
from typing import Dict, Union, Optional

import numpy as np
from pydantic import BaseModel, validator

MIL_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ASC3_DIR = os.path.dirname(MIL_MODEL_DIR)
ROOT_DIR = os.path.dirname(ASC3_DIR)
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "data", "checkpoint")
sys.path.append(ROOT_DIR)
from ASC3.tree_model.model import convert_ad_to_vaf
from core.snv_factory import convert_rule_strength_to_vec, parse_acmg


class SNVFeature(BaseModel):
    """
    3ASC variant spec.

    Note:
        - "ad"가 "." 또는 ","를 구분자로 split될 수 있는지 validation 수행
        - 모든 필드가 required.
    """

    acmg_bayesian: float
    qual: float
    ad: str
    dp: float
    disease_similarity: float
    inhouse_variant_ac: int
    is_incomplete_zygosity: int
    gnomad_gene_pLI: Optional[float] = None
    gnomad_gene_loeuf: Optional[float] = None
    splice_ai_value: Optional[float] = None
    gnomad_AC: Optional[float] = None
    clinvar_variant_scv_pathogenicity_n_p: Optional[float] = None
    clinvar_variant_scv_pathogenicity_n_b: Optional[float] = None
    rule: Optional[str] = None
    header: list = list()

    def check_ad_format(self, ad: str):
        if len(ad.split(".")) != 2 and len(ad.split(",")) != 2:
            raise ValueError(f"""expected ad format: "123.4", incomming ad:{ad}""")
        return ad

    def __post_init__(self):
        self.check_ad_format(self.ad)

    @validator("gnomad_gene_pLI", pre=True, always=True)
    def modify_gnomad_gene_pLI(cls, gnomad_gene_pLI):
        if gnomad_gene_pLI is None:
            return -1
        else:
            return gnomad_gene_pLI

    @validator("gnomad_gene_loeuf", pre=True, always=True)
    def modify_gnomad_gene_loeuf(cls, gnomad_gene_loeuf):
        if gnomad_gene_loeuf is None:
            return -1
        else:
            return gnomad_gene_loeuf

    @validator("splice_ai_value", pre=True, always=True)
    def modify_splice_ai_value(cls, splice_ai_value):
        if splice_ai_value is None:
            return -1
        else:
            return splice_ai_value

    @validator("gnomad_AC", pre=True, always=True)
    def modify_gnomad_AC(cls, gnomad_AC):
        if gnomad_AC is None:
            return 0
        else:
            return gnomad_AC

    @validator("clinvar_variant_scv_pathogenicity_n_p", pre=True, always=True)
    def modify_clinvar_variant_scv_pathogenicity_n_p(cls, clinvar_variant_scv_pathogenicity_n_p):
        if clinvar_variant_scv_pathogenicity_n_p is None:
            return 0
        else:
            return clinvar_variant_scv_pathogenicity_n_p

    @validator("clinvar_variant_scv_pathogenicity_n_b", pre=True, always=True)
    def modify_clinvar_variant_scv_pathogenicity_n_b(cls, clinvar_variant_scv_pathogenicity_n_b):
        if clinvar_variant_scv_pathogenicity_n_b is None:
            return 0
        else:
            return clinvar_variant_scv_pathogenicity_n_b

    @validator("rule", pre=True, always=True)
    def modify_rule(cls, rule) -> str:
        if rule is None:
            return ""
        else:
            return rule

    def to_vector(self, inhouse_total_ac: int) -> np.ndarray:
        vaf = convert_ad_to_vaf(self.ad, delimiter=".")
        inhouse_af = self.inhouse_variant_ac / inhouse_total_ac
        assigned_acmg: dict = parse_acmg(self.rule)
        rule_vector = [num for num in convert_rule_strength_to_vec(assigned_acmg)]

        return np.array(
            [
                self.acmg_bayesian,
                self.disease_similarity,
                self.qual,
                inhouse_af,
                vaf,
                self.is_incomplete_zygosity,
                self.gnomad_gene_pLI,
                self.gnomad_gene_loeuf,
                self.splice_ai_value,
                self.gnomad_AC,
                self.clinvar_variant_scv_pathogenicity_n_p,
                self.clinvar_variant_scv_pathogenicity_n_b,
            ]
            + rule_vector
        )


class CNVFeature(BaseModel):
    acmg_bayesian: float
    disease_similarity: float
    num_genes: int


class MILRequest(BaseModel):
    """
    Request model for 3ASC server.
    """

    sample_id: str
    inhouse_total_ac: int
    snv: Dict[str, Dict[str, SNVFeature]]
    cnv: Union[Dict[str, CNVFeature], Dict]

    class Config:
        schema_extra = {
            "example": {
                "sample_id": "ETA24-ABCD",
                "inhouse_total_ac": 100,
                "snv": {
                    "OMIM:600276-OMIM:125310": {
                        "1-100-A-T": {
                            "acmg_bayesian": 0.9999,
                            "qual": 1271.77,
                            "ad": "287.27",
                            "dp": 129.0,
                            "disease_similarity": 3.09534946,
                            "inhouse_variant_ac": 237,
                            "is_incomplete_zygosity": 1,
                            "gnomad_gene_pLI": 1.3,
                            "gnomad_gene_loeuf": 1.3,
                            "splice_ai_value": 1.3,
                            "gnomad_AC": 1,
                            "clinvar_variant_scv_pathogenicity_n_p": 1.3,
                            "clinvar_variant_scv_pathogenicity_n_b": 1.3,
                            "rule": "PVS1_VS||PM1_M||BS2_S",
                        },
                        "1-200-A-T": {
                            "acmg_bayesian": 0.99409,
                            "qual": 1271.77,
                            "ad": "287.27",
                            "dp": 129.0,
                            "disease_similarity": 5.09534946,
                            "inhouse_variant_ac": 3,
                            "is_incomplete_zygosity": 1,
                            "gnomad_gene_pLI": None,
                            "gnomad_gene_loeuf": 1.3,
                            "splice_ai_value": 1.3,
                            "gnomad_AC": 1,
                            "clinvar_variant_scv_pathogenicity_n_p": 1.3,
                            "clinvar_variant_scv_pathogenicity_n_b": 1.3,
                            "rule": "PM2_M",
                        },
                    },
                    "OMIM:12367-OMIM:23223": {
                        "3-100-A-T": {
                            "acmg_bayesian": 0.99409,
                            "qual": 1271.77,
                            "ad": "287.27",
                            "dp": 129.0,
                            "disease_similarity": 1.09534946,
                            "inhouse_variant_ac": 237,
                            "is_incomplete_zygosity": 0,
                            "gnomad_gene_pLI": 1.3,
                            "gnomad_gene_loeuf": 1.3,
                            "splice_ai_value": 1.3,
                            "gnomad_AC": 1,
                            "clinvar_variant_scv_pathogenicity_n_p": 1.3,
                            "clinvar_variant_scv_pathogenicity_n_b": 1.3,
                            "rule": "BS2_S",
                        },
                    },
                },
                "cnv": {
                    "3-273823-2998433": {
                        "acmg_bayesian": 1.3,
                        "disease_similarity": 4.2,
                        "num_genes": 8,
                    },
                },
            }
        }


class SampleId(BaseModel):
    sample_id: str
