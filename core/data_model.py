from __future__ import annotations

import numbers
from datetime import datetime
from typing import Tuple, List, Iterable, Literal, Union, Any
from dataclasses import dataclass, field, asdict, replace

import numpy as np


@dataclass
class Variant:
    """
    Data class representing a genetic variant with associated information.

    Attributes:
        cpra (str): The CPRA (Chromosome-Position-Reference-Allele) identifier.
        acmg_rules (List[str]): A list of ACMG (American College of Medical Genetics)
                                rules associated with the variant.
        disease_id (str): The identifier for the associated disease (optional).
        gene_id (str): The identifier for the associated gene (optional).

        sample_id (str): The identifier for the sample (optional).
        score (float): The score predcited by 3asc, -1 indicates values is missing (default: -1.0).
        variant_type (str): The type of the variant (optional).

    Methods:
        __eq__(self, other: Any) -> bool:
            Compare two Variant objects for equality based on cpra, acmg_rules, and disease_id.

        to_dict(self) -> dict:
            Convert the Variant object to a dictionary with 'acmg_rules' joined as a string.

    """

    cpra: str
    acmg_rules: List[str] = field(default_factory=list)
    disease_id: str = ""
    gene_id: str = ""
    symbol: str = ""

    sample_id: str = ""
    score: float = -1.0
    variant_type: str = ""

    def __eq__(self, other: Any) -> bool:
        """
        Note:
            symbol은 HGNC에서 계속 변할 수 있어서 eq비교를 안하는 것이
            용이할 듯.
        """
        try:
            return (
                self.cpra == other.cpra
                and self.acmg_rules == other.acmg_rules
                and self.disease_id == other.disease_id
            )

        except AttributeError:
            return False

    def to_dict(self) -> dict:
        """
        Convert the Variant object to a dictionary.

        Returns:
            dict: A dictionary representation of the Variant object with 'acmg_rules' joined as a string.
        """
        return asdict(replace(self, acmg_rules="||".join(self.acmg_rules)))


@dataclass
class Report:
    """환자의 레포트 정보를 저장하는 데이터 클레스

    Note:
        Tucuxi-Aanalysis 내 PK:sample_id, SK:REPORT 에 해당하는 엔티티
    """

    sample_id: str
    conclusion: Literal["positive", "inconclusive", "negative", ""] = str()
    snv_variant: List[Variant] = field(default_factory=list)
    cnv_variant: List[Variant] = field(default_factory=list)
    report_date: datetime = datetime.min


@dataclass
class SNVData:
    x: np.array = np.array([])
    y: np.array = np.array([])

    tsv_path: str = ""
    header: List[str] = field(default_factory=list)
    variants: List[Variant] = field(default_factory=list)
    causal_variant: List[Tuple[str, str]] = field(default_factory=list)  # CPRA, OMIM

    @property
    def n_variants(self):
        return len(self.x)

    def __repr__(self) -> str:
        return f"SNVData(n_variants={self.n_variants}, causal_variant={self.causal_variant})"


@dataclass
class CNVData:
    causal_variant: List[str] = field(default_factory=list)  # position
    x: np.array = np.array([])
    y: np.array = np.array([])
    variants: List[Variant] = field(default_factory=list)
    header: List[str] = field(default_factory=list)

    @property
    def n_variants(self):
        return len(self.x)

    def __repr__(self) -> str:
        return f"CNVData(n_variants={self.n_variants}, causal_variant={self.causal_variant})"


@dataclass
class Symptom:
    """schema originally from OrderManagement tabel"""

    hpo: str
    title: str
    onsetAge: str
    # "Adolescent', 'Adult', 'Antenatal', 'Childhood',
    #'Elderly', 'Infancy', 'Neonatal', 'Unknown'


@dataclass
class PatientData:
    sample_id: str
    bag_label: bool
    snv_data: SNVData = field(default_factory=SNVData)
    cnv_data: CNVData = field(default_factory=CNVData)
    conclusion: Literal["positive", "inconclusive", "negative", ""] = str()

    # private fields
    __symptoms: List[Symptom] = field(init=False, repr=False)
    __gender: Literal = field(init=False, repr=False)

    @property
    def symptoms(self):
        return self.__symptoms

    @symptoms.setter
    def symptoms(self, val: List[Symptom]):
        """
        Example:
            >>> db = DynamoDBClient(...)
            >>> result = db.get_clinical_info(p_data.sample_id)
            >>> p_data.symptoms = [Symptom(*symptom) for symptom in result["symptoms"]]
            >>> print(p_data.symptoms)
            [Symptom(hpo='HP:0003270', title='Abdominal distension', onsetAge='Infancy'), ...]
        """
        if not isinstance(val, list):
            raise ValueError(
                f"Given input type({type(val)}) is not supported, only List[Symptom]"
                + f" is supported."
            )
        self.__symptoms = val

    @property
    def gender(self):
        return self.__gender

    @gender.setter
    def gender(self, val: str):
        """
        Example:
            >>> db = DynamoDBClient(...)
            >>> result = db.get_clinical_info(p_data.sample_id)
            >>> p_data.gender = result["gender"]
            >>> print(p_data.gender)
            "male"
        """
        self.__gender = val

    @property
    def all_hpos(self):
        return [x.hpo for x in self.symptoms]

    @property
    def all_hpos_with_onset(self):
        return [(x.hpo, x.onsetAge) for x in self.symptoms]

    def __repr__(self) -> str:
        return (
            f"PatientData(sample_id={self.sample_id}, bag_label={self.bag_label}, "
            + f"n_snv={self.snv_data.n_variants}, n_cnv={self.cnv_data.n_variants})"
        )


@dataclass
class MetaData:
    created: str = ""  # "%y-%m-%d"

    snv_header: List[str] = field(default_factory=list)
    cnv_header: List[str] = field(default_factory=list)

    n_snv_features: int = 0
    n_cnv_features: int = 0

    def __repr__(self) -> str:
        return (
            f"MetaData(created={self.created}, n_snv_features={self.n_snv_features}, "
            + f"n_cnv_features={self.n_cnv_features})"
        )

    def to_dict(self):
        return self.__dict__


@dataclass
class PatientDataSet:
    """
    Represents a dataset containing patient data, including SNV and CNV features.

    Attributes:
        data (List[PatientData]): A list of PatientData instances representing patient records.
        metadata (MetaData): Metadata associated with the dataset, automatically generated during initialization.

    Methods:
        __post_init__(): Post-initialization method to set up the metadata based on the input data.

    Example:
        # Create a PatientDataSet instance with some PatientData objects
        >>> patient_dataset = PatientDataSet(data=[PatientData(...), PatientData(...), PatientData(...)])

        # Access the metadata associated with the dataset
        >>> patient_dataset.metadata.snv_header
        ['ACMG_bayesian', 'symptom_similarity', 'vcf_info_QUAL', 'inhouse_freq', 'vaf', 'is_incomplete_zygosity']

        >>> patient_dataset["ETA23-ABCD"]
        PatientData(sample_id="ETA23-ABCD")

        >>> patient_dataset[0]
        PatientData(sample_id="ETA23-ABCD")

        >>> patient_dataset[[0, 1, 2]]
        PatientDataSet(len=3)

        >>> PatientDataSet(data=[PatientData(...), [PatientData(...)])
        PatientDataSet(len=3)
        >>> PatientDataSet([PatientData(...), [PatientData(...)])
        PatientDataSet(len=3)
    """

    data: List[PatientData] = field(default_factory=list)
    metadata: MetaData = field(default_factory=MetaData)

    _data: List[PatientData] = field(init=False, repr=False)

    def __post_init__(self):
        snv_header = list()
        cnv_header = list()
        n_snv_features = 0
        n_cnv_features = 0

        for patient_data in self.data:
            if len(patient_data.snv_data.x) == 0:
                continue

            snv_header = patient_data.snv_data.header
            n_snv_features = patient_data.snv_data.x.shape[1]
            break

        for patient_data in self.data:
            if len(patient_data.cnv_data.x) == 0:
                continue

            cnv_header = patient_data.cnv_data.header
            n_cnv_features = patient_data.cnv_data.x.shape[1]
            break

        arg_dict = {
            "created": datetime.today().strftime("%y-%m-%d"),
            "snv_header": snv_header,
            "cnv_header": cnv_header,
            "n_snv_features": n_snv_features,
            "n_cnv_features": n_cnv_features,
        }

        self._data = self.data
        self.metadata = MetaData(**arg_dict)

        self.sample_id_to_idx = {
            p_data.sample_id: i for i, p_data in enumerate(self.data)
        }

    def __repr__(self) -> str:
        return f"PatientDataSet(len={self.__len__()}, created={self.metadata.created})"

    def __getitem__(
        self, idx: Union[str, int, Iterable[str], Iterable[int]]
    ) -> Union[PatientData, PatientDataSet]:
        if not hasattr(self, "sample_id_to_idx"):
            self.sample_id_to_idx = {
                p_data.sample_id: i for i, p_data in enumerate(self.data)
            }

        if isinstance(idx, numbers.Integral):
            return self.data[idx]

        elif isinstance(idx, str):
            if idx not in self.sample_id_to_idx:
                raise IndexError(f"Passed sample_id({idx}) is not found.")
            return self.data[self.sample_id_to_idx[idx]]

        elif isinstance(idx, Iterable):
            if all([isinstance(x, str) for x in idx]):
                sample_ids = idx
                subset = list()
                for sample_id in sample_ids:
                    if sample_id not in self.sample_id_to_idx:
                        raise IndexError(f"Passed sample_id({str}) not found.")
                    subset.append(self.data[self.sample_id_to_idx[sample_id]])
                return PatientDataSet(subset)

            elif all([isinstance(x, numbers.Integral) for x in idx]):
                return PatientDataSet([self.data[i] for i in idx])

        return PatientDataSet(self.data[idx])

    def __contains__(self, sample_id):
        return sample_id in self.all_sample_ids

    def __len__(self):
        return len(self.data)

    def get_snv_header(self):
        return self.metadata.snv_header

    def get_cnv_header(self):
        return self.metadata.cnv_header

    # TODO
    def set_scaler(self):
        pass

    # TODO
    def scale_data(self, scaler=None):
        if not hasattr(self, "scaler") and scaler is None:
            self.set_scaler()
        pass

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data: List[PatientData]):
        self._data = new_data
        self.sample_id_to_idx = {
            p_data.sample_id: i for i, p_data in enumerate(new_data)
        }

    @property
    def snv_x(self) -> np.ndarray:
        """Get Single Nucleotide Variant (SNV) data as a NumPy array.

        Note:
            If there is no data in the dataset, an empty array will be returned.

        Returns:
            np.ndarray: A 2D NumPy array representing the SNV data.

        Example:
            >>> dataset = PatientDataSet(data=[PatientData(...), PatientData(...), ...])
            >>> dataset.snv_x
            array([[0.26899999380111694, -1.0, -1.0, ..., 0, 0.12831858407079647, False],
                    [-1.0, -1.0, -1.0, ..., 0, 0.20714285714285716, False],
                    [-1.0, -1.0, -1.0, ..., 0, 0.07990867579908675, False],
                    ...,
                    [0.9229999780654907, -1.0, -1.0, ..., 0, 0.43137254901960786, False],
                    [-1.0, -1.0, -1.0, ..., 0, 0.5, False],
                    [-1.0, -1.0, -1.0, ..., 0, 0.4873417721518987, False]], dtype=object)
        """
        if len(self.data) == 0:
            n_dim = len(self.get_snv_header())
            return np.empty(shape=(0, n_dim))
        return np.concatenate([data.snv_data.x for data in self.data])

    @property
    def cnv_x(self) -> np.ndarray:
        """Get Copy Number Variant (CNV) data as a NumPy array.

        Note:
            If there is no data in the dataset, an empty array will be returned.

        Returns:
            np.ndarray: A 2D NumPy array representing the CNV data.

        Examples:
            >>> dataset = PatientDataSet(data=[PatientData(...), PatientData(...), ...])
            >>> dataset.cnv_x
            array([[ 0.      ,  0.      , 10.      ],
                    [ 0.      ,  0.      ,  3.      ],
                    [ 1.4     ,  6.357143, 21.      ],
                    ...,
                    [ 0.      ,  0.      , 19.      ],
                    [ 0.      ,  3.6875  ,  2.      ],
                    [-1.      , -1.      , -1.      ]], dtype=float32)
        """
        if len(self.data) == 0:
            n_dim = len(self.get_cnv_header())
            return np.empty(shape=(0, n_dim))
        return np.concatenate([data.cnv_data.x for data in self.data])

    @property
    def bag_labels(self):
        if len(self.data) == 0:
            return np.empty(shape=(0,))
        return np.array([data.bag_label for data in self.data], dtype=np.float32)

    @property
    def all_sample_ids(self):
        if len(self.data) == 0:
            return np.empty(shape=(0,))
        return np.array([data.sample_id for data in self.data])

    @property
    def snv_causal_var_x(self):
        if len(self.data) == 0:
            return np.empty(shape=(0, 6))

        causal_xs = []
        for data in self.data:
            causal_in_patient = data.snv_data.x[np.where(data.snv_data.y == 1)]
            if len(causal_in_patient) > 0:
                causal_xs.append(causal_in_patient)
        return np.concatenate(causal_xs)

    @property
    def snv_non_causal_var_x(self):
        if len(self.data) == 0:
            return np.empty(shape=(0, 6))

        non_causal_xs = []
        for data in self.data:
            non_causal_in_patient = data.snv_data.x[np.where(data.snv_data.y == 0)]
            if len(non_causal_in_patient) > 0:
                non_causal_xs.append(non_causal_in_patient)

        return np.concatenate(non_causal_xs)

    @property
    def cnv_causal_var_x(self):
        if len(self.data) == 0:
            return np.empty(shape=(0, 3))

        causal_xs = []
        for data in self.data:
            causal_in_patient = data.cnv_data.x[np.where(data.cnv_data.y == 1)]
            if len(causal_in_patient) > 0:
                causal_xs.append(causal_in_patient)

        return np.concatenate(causal_xs)

    @property
    def cnv_non_causal_var_x(self):
        if len(self.data) == 0:
            return np.empty(shape=(0, 3))

        non_causal_xs = []
        for data in self.data:
            non_causal_in_patient = data.cnv_data.x[np.where(data.cnv_data.y == 0)]
            if len(non_causal_in_patient) > 0:
                non_causal_xs.append(non_causal_in_patient)
        return np.concatenate(non_causal_xs)

    @property
    def snv_instance_y(self):
        if len(self.data) == 0:
            return np.empty(shape=(0,))
        return np.concatenate([data.snv_data.y for data in self.data])

    @property
    def cnv_instance_y(self):
        if len(self.data) == 0:
            return np.empty(shape=(0,))
        return np.concatenate([data.cnv_data.y for data in self.data])

    @property
    def snv_causal_variant(self):
        return [
            (p_data.sample_id, p_data.snv_data.causal_variant)
            for p_data in self.data
            if len(p_data.snv_data.causal_variant) > 0
        ]

    @property
    def cnv_causal_variant(self):
        return [
            (p_data.sample_id, p_data.cnv_data.causal_variant)
            for p_data in self.data
            if len(p_data.cnv_data.causal_variant) > 0
        ]
