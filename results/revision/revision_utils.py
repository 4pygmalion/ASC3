import mlflow

from mlflow.entities import Run
from typing import List
from mlflow.tracking import MlflowClient

TRACKING_URI = ""

RUN_ASC3_W_RANKNET = "29f1dfc8ee2b464cb6ec36627f12e429"
RUN_ASC3_WO_RANKNET = "87cec01ac82a42a6a6fffce3b7543597"
FOLD_RESULTS_PICKLE = "mlflow-artifacts:/6/{run_id}/artifacts/fold_result.pickle"


def open_pickle(path):
    import pickle

    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def get_child_runs(parent_run_id) -> List[Run]:
    client = MlflowClient()

    all_runs = client.search_runs(
        experiment_ids=[6],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
    )

    child_runs = []
    for run in all_runs:
        child_runs.append(run.info.run_id)

    return child_runs
