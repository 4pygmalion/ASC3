"""LIRICAL 결과를 파싱하여 변이 또는 유전자별로 파싱하는 모듈

// tree -l {sample_id}
{sample_id}
├── lirical-config.yaml
├── SH254-596.html
├── SH254-596.symbol.tsv # gene annotatated
└── SH254-596.tsv # original file

// {sample_id}.symbol.tsv의 파일포맷
rank	diseaseName	diseaseCurie	pretestprob	posttestprob	compositeLR	entrezGeneId	variants	symbol
1	Raine syndrome	OMIM:259775	1/8311	5.58%	491.505	NCBIGene:56975	chr7:286468G>GGACAGGTGAGCCCTTCCTTCCTCCCTCCATCCGC uc003sip.3:c.957_958insTGAGCCCTTCCTTCCTCCCTCCATCCGCGACAGG:p.(Ile320*) pathogenicity:1.0 [HOMOZYGOUS_ALT]; chr7:299850G>T uc003sip.3:c.1659G>T:p.(=) pathogenicity:0.0 [HETEROZYGOUS]; chr7:299881A>G uc003sip.3:c.1690A>G:p.(Asn564Asp) pathogenicity:0.0 [HETEROZYGOUS]	FAM20C
2	HYDROXYKYNURENINURIA	OMIM:236800	1/8311	0.14%	11.455	NCBIGene:8942	chr2:143713800G>A uc002tvk.3:c.464G>A:p.(Arg155Gln) pathogenicity:1.0 [HETEROZYGOUS]; chr2:143798189A>G uc002tvl.3:c.1234A>G:p.(Lys412Glu) pathogenicity:0.0 [HETEROZYGOUS]	KYNU

Example:
    >>> import tqdm
    >>> from a4c.utils.parse_lirical_files import LiricalOutputParser

    >>> lirical_dir = exomiser_dir.replace("exomiser", "lirical")
    >>> lirical_dfs = dict()
    >>> lirical = LiricalOutputParser(lirical_dir)
    >>> for sample_id in tqdm.tqdm(os.listdir(lirical_dir)):
            try:
                lirical_dfs[sample_id] = lirical.parse(sample_id)
            except:
                continue
"""

import os
import gzip
from typing import List, Tuple, Dict
from collections import defaultdict

import pandas as pd


class LiricalOutputParser:
    def __init__(self, root_dir: str, mode="variants") -> None:
        """LIRICAL 출력포맷을 파싱하는 클레스

        Args:
            root_dir (str): 모든 샘플의 LIRICAL 결과가 담긴 폴더
            mode (str, optional): "genes" or "variant". Defaults to "variants".
        """
        self.root_dir = root_dir
        self.mode = mode
        if mode == "genes":
            self.set_gene2symbol()

    def set_gene2symbol(self):
        gene2symbols = defaultdict(set)

        with gzip.open(
            "/DAS/data/DB/processedData/RefSeq/backup/20230510/merged.refseq.grch37.all.txt.gz",
            "rt",
        ) as fr:
            col2idx = dict()
            for line in fr:
                row = line.lstrip("#").strip().split("\t")
                if line.startswith("#"):
                    col2idx = {col: idx for idx, col in enumerate(row)}
                    continue

                symbols = row[col2idx["gene_symbol"]].split("||")
                for gene_id in row[col2idx["gene_id"]].split("||"):
                    for symbol in symbols:
                        gene2symbols[gene_id] = symbol

        self.gene2symbol = gene2symbols

        return

    def preprocess_cpra(self, raw_cpra: str) -> str:
        """LIRICAL에서 반환하는 변이명의 포맷팅을 C-P-R-A 형식으로 변환함

        Args:
            raw_cpra (str): raw_cpra

        Returns:
            str: preprocessed_cpra

        Example:
            >>> self.preprocess_cpra("chr9:2039776ACAGCAGCAGCAGCAG>A")
            "9-2039776-ACAGCAGCAGCAGCAG-A"
        """

        chrom, remnant = raw_cpra.split(":", maxsplit=1)
        position_end_idx = 0
        for idx, char in enumerate(remnant):
            if not char.isdigit():
                position_end_idx = idx
                break

        pos_ref, alt = remnant.split(">")
        pos, ref = (
            pos_ref[:position_end_idx],
            pos_ref[position_end_idx:],
        )
        return chrom.replace("chr", "") + f"-{pos}-{ref}-{alt}"

    def _split_tokens(self, variants_field_value: str) -> List[Tuple[str, float]]:
        """필드값(tokens)을 분리하여 변이-점수의 리스트로 반환

        Args:
            variants_field_value (str): 변이-점수의 필드값

        Returns:
            List[Tuple[str, float]]: 변이별 점수의 리스트

        Example:
            >>> variants_field_value = (
                "chr13:78477732T>C uc001vko.2:c.494A>G:p.(Glu165Gly) pathogenicity:0.8 [HETEROZYGOUS]; "
                "chr13:78477674A>G uc001vko.2:c.552C>C:p.(=) pathogenicity:0.0 [HOMOZYGOUS_ALT]"
            )
            >>> self._split_tokens(variants_field_value)
            [{"13-78477732-T-C": 0.8}, {"13-78477674-A-G": 0.0}]
        """

        pairs = list()
        for variant_score in variants_field_value.split(";"):
            variant_score = variant_score.strip()
            if variant_score == "n/a":
                continue

            (
                raw_cpra,
                _desc,
                raw_pathogenicity,
                _inheritance_pattern,
            ) = variant_score.split(" ")

            cpra = self.preprocess_cpra(raw_cpra)
            pathogenicity = float(raw_pathogenicity.split(":")[-1])
            pairs.append((cpra, pathogenicity))

        return pairs

    def parse_variant_file(self, sample_id) -> Dict[str, float]:
        sample_dir = os.path.join(self.root_dir, sample_id)

        columns = list()
        col2idx = dict()
        data = defaultdict(float)
        with open(os.path.join(sample_dir, f"{sample_id}.tsv")) as fh:
            for idx, line in enumerate(fh):
                if line.startswith("!"):
                    continue

                columns = line.strip().split("\t")
                col2idx = {col: idx for idx, col in enumerate(columns)}
                break

            for idx, row in enumerate(fh):
                row = row.strip().split("\t")
                cpra_score_pairs = self._split_tokens(row[col2idx["variants"]])
                for cpra, score in cpra_score_pairs:
                    data[cpra] = max(data[cpra], score)

        return data

    def parse_gene_file(self, sample_id):
        sample_dir = os.path.join(self.root_dir, sample_id)

        data = defaultdict(float)
        with open(os.path.join(sample_dir, f"{sample_id}.tsv")) as fh:
            for idx, line in enumerate(fh):
                if line.startswith("!"):
                    continue

                (
                    _remaining,
                    posttestprob,
                    compositeLR,
                    entrez_gene_id,
                    tokens,
                ) = line.rsplit("\t", maxsplit=4)

                gene_id = entrez_gene_id.lstrip("NCBIGene:")
                if gene_id not in self.gene2symbol:
                    data["-"] = 0
                    continue

                symbol = self.gene2symbol[gene_id]

                data[symbol] = max(
                    data[symbol],
                    float(posttestprob[:-1]) / 100,  # % to float
                )
        return data

    def parse(self, sample_id):
        if self.mode == "variants":
            data = self.parse_variant_file(sample_id)
        else:
            data = self.parse_gene_file(sample_id)

        if not data:
            return pd.DataFrame()

        item_col, score_col = zip(*data.items())
        item_colname = "cpra" if self.mode == "variants" else "symbol"

        return pd.DataFrame(
            data={item_colname: item_col, "score": score_col},
        )
