"""Exomiser을 실행

Example:
    // exomiser directory로 이동
    $ cd /data2/heon_dev/repository/exomiser/exomiser-cli-13.0.1 

    // 다음의 코드를 실행
    $ python3 /data2/heon_dev/repository/3ASC-Confirmed-variant-Resys/a4c/utils/run_exomiser.py \
        -c /data2/heon_dev/repository/3ASC-Confirmed-variant-Resys/a4c/config.yaml \
        --sample_path /data2/heon_dev/repository/3ASC-Confirmed-variant-Resys/notebooks/MIL/figures/test_sample_ids.csv \
        --result_dir /data2/heon_dev/repository/3ASC-Confirmed-variant-Resys/notebooks/MIL/exomiser
"""

import os
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, wait

import boto3
from tqdm import tqdm
import pandas as pd
from omegaconf import OmegaConf

BM_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BM_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
from .paths import NAS_VCF_PATH, GDS_VCF_PATH, EXOMISER_PATH


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="config path")
    parser.add_argument(
        "--sample_path",
        type=str,
        help="csv file including sample to run",
        required=True,
    )
    parser.add_argument(
        "--result_dir",
        help="Exomiser result directory",
        type=str,
        default=os.path.join(ROOT_DIR, "Exomiser"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    ARGS = get_args()
    CONFIG = OmegaConf.load(ARGS.config)
    RESULTS_FOLDER = ARGS.result_dir
    SAMPLES_PATH = ARGS.sample_path

    EXOMISER_CONFIG = CONFIG.EXOMISER
    CONFIG_TEMPLATE_PATH = EXOMISER_CONFIG["CONFIG_TEMPLATE_PATH"]
    KEYFILE_PATH = os.path.join(DATA_DIR, EXOMISER_CONFIG["KEYFILE_PATH"])

    os.chdir(EXOMISER_PATH)

    SECRET_CONFIG = OmegaConf.laod(os.path.join(DATA_DIR, KEYFILE_PATH))
    DYNAMODB_CLIENT = boto3.resource(
        "dynamodb",
        region_name="ap-northeast-2",
        aws_access_key_id=SECRET_CONFIG["DYNAMODB"]["ACCESS_KEY"],
        aws_secret_access_key=SECRET_CONFIG["DYNAMODB"]["SECRET_KEY"],
    )

    SAMPLES = list(set(pd.read_csv(SAMPLES_PATH)["sample_id"]))

    no_hpo_count = 0
    execute_count = 0
    futures = list()
    process_pool_executor = ProcessPoolExecutor(max_workers=4)
    for sample in tqdm(SAMPLES):
        result_folder = os.path.join(RESULTS_FOLDER, sample)
        if not os.path.isdir(result_folder):
            os.makedirs(result_folder)

        prefix = sample.split("-")[-1][:2]
        vcf_path = NAS_VCF_PATH.format(prefix=prefix, sample=sample)
        if not os.path.exists(vcf_path):
            vcf_path = GDS_VCF_PATH.format(prefix=prefix, sample=sample)

        if not os.path.exists(vcf_path):
            print(vcf_path, "not found")
            continue

        config_path = os.path.join(result_folder, "exomiser-config.yaml")

        hpos = str()
        response = (
            DYNAMODB_CLIENT.Table("Tucuxi-Analysis")
            .get_item(Key={"PK": sample, "SK": "SYMPTOM"})
            .get("Item", dict())
            .get("hpo", list())
        )
        if response:
            hpos = [hpo_info["hpo"] for hpo_info in response if hpo_info["hpo"] != "-"]
        if not hpos:
            no_hpo_count += 1
            print(f"## No HPO: {sample}")
            print(no_hpo_count)
            continue

        if os.path.isfile(os.path.join(result_folder, f"{sample}_AD.genes.tsv")):
            continue

        execute_count += 1
        print(execute_count)
        with open(CONFIG_TEMPLATE_PATH) as fr, open(config_path, "w") as fw:
            for line in fr:
                if line.strip().startswith("vcf:"):
                    line = line.replace("examples/Pfeiffer.vcf", vcf_path)
                elif line.strip().startswith("hpoIds:"):
                    line = line.replace(
                        "HP:0001156', 'HP:0001363', 'HP:0011304', 'HP:0010055",
                        "', '".join(hpos),
                    )
                elif line.strip().startswith("pathogenicitySources:"):
                    line = line.replace(
                        "POLYPHEN, MUTATION_TASTER, SIFT, REMM",
                        "POLYPHEN, MUTATION_TASTER, SIFT",
                    )
                elif line.strip().startswith("outputPrefix:"):
                    line = line.replace(
                        "results/Pfeiffer-hiphive-exome-PASS_ONLY",
                        os.path.join(result_folder, sample),
                    )
                elif line.strip().startswith("outputFormats:"):
                    line = line.replace(
                        "HTML, JSON, TSV_GENE, TSV_VARIANT, VCF",
                        "HTML, TSV_GENE, TSV_VARIANT",
                    )
                fw.write(line)

        command = (
            "java "
            "-Xms2g -Xmx4g "
            "-jar exomiser-cli-13.0.1.jar "
            f"--analysis {config_path} "
            "--exomiser.data-directory=/data2/heon_dev/repository/exomiser/exomiser-cli-13.0.1/exomiser-data"
        )
        print(command)
        futures.append(
            process_pool_executor.submit(
                subprocess.run, command, shell=True, check=True
            )
        )

    wait(futures)
