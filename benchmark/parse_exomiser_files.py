"""Exomiser의 결과를 파싱하는 모듈

Exomiser output은 다음과 같이 variant단위 및 gene단위로 구성되어있음.
(3asc) $ tree -L 2 sample_dir
    sample_dir
    ├── {sample_dir}_AD.genes.tsv
    ├── {sample_dir}_AD.variants.tsv
    ├── {sample_dir}_AR.genes.tsv
    ├── {sample_dir}_AR.variants.tsv
    ├── {sample_dir}.html
    ├── {sample_dir}_MT.genes.tsv
    ├── {sample_dir}_MT.variants.tsv
    ├── {sample_dir}_XD.genes.tsv
    ├── {sample_dir}_XD.variants.tsv
    ├── {sample_dir}_XR.genes.tsv
    ├── {sample_dir}_XR.variants.tsv
    └── exomiser-config.yaml

// {sample_id}.AD.variant.tsv 포맷
#CHROM	POS	REF	ALT	QUAL	FILTER	GENOTYPE	COVERAGE	FUNCTIONAL_CLASS	HGVS	EXOMISER_GENE	CADD(>0.483)	POLYPHEN(>0.956|>0.446)	MUTATIONTASTER(>0.94)	SIFT(<0.06)	REMM	DBSNP_ID	MAX_FREQUENCY	DBSNP_FREQUENCY	EVS_EA_FREQUENCY	EVS_AA_FREQUENCY	EXAC_AFR_FREQ	EXAC_AMR_FREQ	EXAC_EAS_FREQ	EXAC_FIN_FREQ	EXAC_NFE_FREQ	EXAC_SAS_FREQ	EXAC_OTH_FREQ	EXOMISER_VARIANT_SCORE	EXOMISER_GENE_PHENO_SCORE	EXOMISER_GENE_VARIANT_SCORE	EXOMISER_GENE_COMBINED_SCORE	CONTRIBUTING_VARIANT
2	74757278	C	T	2260.77	PASS	0/1	174	stop_gained	HTRA2:ENST00000258080.3:c.145C>T:p.(Arg49*)	HTRA2	.	.	1.0	.	.	.	0.0	.	.	.	.	.	.	.	.	.	.	1.0	0.9849290832679095	1.0	0.9978357940607986	CONTRIBUTING_VARIANT
12	112369626	T	TA	1893.77	PASS	0/1	137	frameshift_elongation	TMEM116:ENST00000355445.3:c.707dup:p.(Thr237Asnfs*30)	TMEM116	.	.	.	.	.	.	0.0	.	.	.	.	.	.	.	.	.	.	1.0	0.8011030522191112	1.0	0.9855535348217102	CONTRIBUTING_VARIANT

// {sample_id}.AD.genes.tsv 포맷
#GENE_SYMBOL	ENTREZ_GENE_ID	EXOMISER_GENE_PHENO_SCORE	EXOMISER_GENE_VARIANT_SCORE	EXOMISER_GENE_COMBINED_SCORE	HUMAN_PHENO_SCORE	MOUSE_PHENO_SCORE	FISH_PHENO_SCORE	WALKER_SCORE	PHIVE_ALL_SPECIES_SCORE	OMIM_SCORE	MATCHES_CANDIDATE_GENE	HUMAN_PHENO_EVIDENCE	MOUSE_PHENO_EVIDENCE	FISH_PHENO_EVIDENCE	HUMAN_PPI_EVIDENCE	MOUSE_PPI_EVIDENCE	FISH_PPI_EVIDENCE
CEACAM16	388551	0.9560	0.9915	0.9968	0.6430	0.9560	0.0000	0.0000	0.9560	1.0000	0	Deafness, autosomal dominant 4B (OMIM:614614): Hearing impairment (HP:0000365)-Hearing impairment (HP:0000365), 	Hearing impairment (HP:0000365)-nonsyndromic hearing loss (MP:0004749), 				
ACAN	176	0.9515	0.9925	0.9967	0.3910	0.9515	0.0000	0.5300	0.9515	1.0000	0	Short stature and advanced bone age, with or without early-onset osteoarthritis and/or osteochondritis dissecans (OMIM:165800): Flexion contracture (HP:0001371)-Osteochondritis Dissecans (HP:0010886), 	Flexion contracture (HP:0001371)-abnormal cartilage development (MP:0000164), Hearing impairment (HP:0000365)-deafness (MP:0001967), Cataract (HP:0000518)-abnormal organ of Corti morphology (MP:0000042), 			Proximity to HAPLN1 Flexion contracture (HP:0001371)-abnormal cartilage development (MP:0000164), Hearing impairment (HP:0000365)-middle ear ossicle hypoplasia (MP:0030124), Cataract (HP:0000518)-abnormal basicranium morphology (MP:0010029), 	


Example:
    >>> import tqdm
    >>> from a4c.utils.parse_exomiser_files import ExomiserOutputParser

    >>> exomiser_dir = "/data2/heon_dev/repository/3ASC-Confirmed-variant-Resys/notebooks/MIL/exomiser"
    >>> exomiser_dfs = dict()
    >>> exomiser = ExomiserOutputParser(exomiser_dir)
    >>> for sample_id in tqdm.tqdm(os.listdir(exomiser_dir)):
            try:
                exomiser_dfs[sample_id] = exomiser.parse_variant(sample_id)
            except:
                continue
"""

import os
from typing import Dict
from collections import defaultdict

import pandas as pd


class ExomiserOutputParser:
    variant_inheritances = {"AD", "AR", "XD", "XR", "MT"}

    def __init__(self, root_dir: str, mode: str = "variants") -> None:
        """
        Args:
            root_dir (str): 샘플별 Exoimiser 결과가 들어있는 최상위 폴더
            mode (str): ["variants", "genes"]
        """
        self.root_dir = root_dir
        self.mode = mode

    def get_gene_with_score(self, filepath: str) -> Dict[str, float]:
        """유전자별 스코어를 파일(filepath)별로 파싱하여 반환

        Args:
            filepath (str): Exomiser의 유전자정보의 tsv 파일의 경로

        Return:
            gene_with_score (dict): 유전자 별 exomiser 점수의 비교
        """
        gene_with_score = defaultdict(float)
        with open(filepath, "r") as fh:
            for row in fh:
                row = row.strip()
                if row.startswith("#"):
                    continue

                (
                    symbol,
                    _entrez_gene_id,
                    _exomiser_gene_pheno_score,
                    _exomiser_gene_variant_score,
                    exomiser_gene_combined_score,
                    _remaining,
                ) = row.split("\t", maxsplit=5)

                gene_with_score[symbol] = max(
                    gene_with_score[symbol],
                    float(exomiser_gene_combined_score),
                )

        return gene_with_score

    def get_variant_with_score(self, filepath: str) -> pd.DataFrame:
        """변이별 스코어를 파일(filepath)별로 파싱하여 반환

        Args:
            filepath (str): Exomiser의 유전자정보의 tsv 파일의 경로

        Return:
            gene_with_score (dict): 유전자 별 exomiser 점수의 비교
        """
        res = list()
        with open(filepath, "r") as fh:
            for line in fh:
                row = line.strip()
                if row.startswith("#"):
                    col2idx = {col: i for i, col in enumerate(row.strip().split("\t"))}
                    continue

                gene = row[col2idx["EXOMISER_GENE"]]
                chr, pos, ref, alt, _others = row.split("\t", maxsplit=4)
                others, gene_variant_score, gene_combing_score, contributing = row.rsplit(
                    "\t", maxsplit=3
                )
                variant_score = row.rsplit("\t", maxsplit=5)[1]

                cpra = "-".join([chr, pos, ref, alt])
                res.append(
                    [
                        cpra,
                        float(variant_score) if variant_score != "" else 0,
                        float(gene_combing_score) if gene_combing_score != "" else 0,
                    ]
                )

        return pd.DataFrame(res, columns=["cpra", "variant_score", "gene_score"])

    def parse_variant(self, sample_id: str) -> pd.DataFrame:
        sample_dir = os.path.join(self.root_dir, sample_id)

        res = list()
        all_inheritance_variant: Dict[str, Dict[str, float]] = dict()
        for inheritance_pattern in self.variant_inheritances:
            filename = f"{sample_id}_{inheritance_pattern}.{self.mode}.tsv"
            file_abs_path = os.path.join(sample_dir, filename)

            variant_df = self.get_variant_with_score(file_abs_path)
            res.append(variant_df)

        return (
            pd.concat(res)
            .sort_values(["gene_score", "variant_score"], ascending=False)
            .drop_duplicates("cpra")
        )

    def collate_maximum(
        self, all_inheritance_variant: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """변이별 inheritance의 별로 나온 결과파일을 집계하여 반환함

        Args:
            all_inheritance_variant (dict): 모든 유전패턴별 변이/유전자별 스코어

        Returns:
            dict: 변이/유전자별 스코어
        """
        union_key = set()
        for inheritance_pattern in self.variant_inheritances:
            union_key |= all_inheritance_variant[inheritance_pattern].keys()

        merged = dict()
        for key in union_key:
            highest = max(
                all_inheritance_variant[pattern].get(key, 0.0)
                for pattern in self.variant_inheritances
            )
            merged[key] = highest

        return merged

    def parse(self, sample_id: str) -> pd.DataFrame:
        sample_dir = os.path.join(self.root_dir, sample_id)

        all_inheritance_variant: Dict[str, Dict[str, float]] = dict()
        for inheritance_pattern in self.variant_inheritances:
            filename = f"{sample_id}_{inheritance_pattern}.{self.mode}.tsv"
            file_abs_path = os.path.join(sample_dir, filename)

            if self.mode == "variants":
                var_dict = self.get_variant_with_score(file_abs_path)
            else:
                var_dict = self.get_gene_with_score(file_abs_path)

            all_inheritance_variant[inheritance_pattern] = var_dict

        maximum_variant_score = self.collate_maximum(all_inheritance_variant)

        item_col, score_col = zip(*maximum_variant_score.items())
        item_colname = "cpra" if self.mode == "variants" else "symbol"
        return pd.DataFrame(data={item_colname: item_col, "score": score_col})
