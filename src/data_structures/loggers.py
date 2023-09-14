import logging
from pathlib import Path
import polars as pl
import datetime as dt
import json
import copy
import string
import random
import os

from time import process_time

RUN_ID_PATH = Path("results/run_id")
SCENARIO_ID_PATH = Path("results/scenario_id")

logging.basicConfig(
    format="%(asctime)s %(message)s",
    filemode="a",
    filename="data/logging.txt",
    level=logging.DEBUG,
)


def setup_run_logging():
    alphabet = string.ascii_lowercase + string.digits
    run_name = "".join(random.choices(alphabet, k=8))
    os.makedirs(f"results/json/{run_name}")
    os.makedirs(f"results/logs/{run_name}")
    os.makedirs(f"results/logs/{run_name}/run_logs")
    os.makedirs(f"results/logs/{run_name}/candidate_logs")

    return run_name


class ScenarioLogger:
    def __init__(
        self,
        source_table,
        git_hash,
        iterations,
        join_strategy,
        aggregation,
        target_dl,
        n_splits,
        top_k,
        feature_selection,
        model_selection,
        run_name=None,
    ) -> None:
        self.timestamps = {
            "start_process": dt.datetime.now(),
            "end_process": 0,
            "start_load_index": 0,
            "end_load_index": 0,
            "start_querying": 0,
            "end_querying": 0,
            "start_evaluation": 0,
            "end_evaluation": 0,
        }
        self.run_name = run_name
        self.scenario_id = self.find_latest_scenario_id(run_name)
        self.prepare_logger(run_name)
        self.run_id = 0
        self.start_timestamp = None
        self.end_timestamp = None
        self.source_table = source_table
        self.git_hash = git_hash
        self.iterations = iterations
        self.join_strategy = join_strategy
        if join_strategy == "nojoin":
            self.aggregation = "nojoin"
        else:
            self.aggregation = aggregation
        self.target_dl = target_dl
        self.n_splits = n_splits
        self.model_selection = model_selection
        self.feature_selection = feature_selection
        self.top_k = top_k
        self.results = None
        self.process_time = 0

    def prepare_logger(self, run_name=None):
        self.path_run_logs = f"results/logs/{run_name}/run_logs/{self.scenario_id}.log"
        self.path_candidate_logs = (
            f"results/logs/{run_name}/candidate_logs/{self.scenario_id}.log"
        )

    def add_timestamp(self, which_ts):
        self.timestamps[which_ts] = dt.datetime.now()

    def add_process_time(self):
        self.process_time = process_time()

    def find_latest_scenario_id(self, run_name=None):
        """Utility function for opening the scenario_id file, checking for errors and
        incrementing it by one at the start of a run.

        Raises:
            ValueError: Raise ValueError if the read scenario_id is not a positive integer.

        Returns:
            int: The new (incremented) scenario_id.
        """
        if run_name is not None:
            scenario_id_path = Path(f"results/json/{run_name}/scenario_id")
        else:
            scenario_id_path = SCENARIO_ID_PATH
        if scenario_id_path.exists():
            with open(scenario_id_path, "r") as fp:
                last_scenario_id = fp.read().strip()
                if len(last_scenario_id) != 0:
                    try:
                        scenario_id = int(last_scenario_id) + 1
                    except ValueError:
                        raise ValueError(
                            f"Scenario ID {last_scenario_id} is not a positive integer. "
                        )
                    if scenario_id < 0:
                        raise ValueError(
                            f"Scenario ID {scenario_id} is not a positive integer. "
                        )
                else:
                    scenario_id = 0
            with open(scenario_id_path, "w") as fp:
                fp.write(f"{scenario_id}")
        else:
            scenario_id = 0
            with open(scenario_id_path, "w") as fp:
                fp.write(f"{scenario_id}")
        return scenario_id

    def get_parameters(self):
        return {
            "source_table": self.source_table,
            "iterations": self.iterations,
            "join_strategy": self.join_strategy,
            "aggregation": self.aggregation,
            "target_dl": self.target_dl,
            "n_splits": self.n_splits,
            "model_selection": self.model_selection,
            "feature_selection": self.feature_selection,
            "top_k": self.top_k,
        }

    def set_results(self, results: pl.DataFrame):
        self.results = results

    def get_next_run_id(self):
        self.run_id += 1
        return self.run_id

    def to_string(self):
        str_res = ",".join(
            map(
                str,
                [
                    self.scenario_id,
                    self.git_hash,
                    self.source_table,
                    self.iterations,
                    self.join_strategy,
                    self.aggregation,
                    self.target_dl,
                    self.n_splits,
                    self.top_k,
                    self.feature_selection,
                    self.model_selection,
                ],
            )
        )
        str_res += ","
        for ts in self.timestamps.values():
            str_res += str(ts) + ","

        return str_res.rstrip(",")

    def pretty_print(self):
        print(f"Run name: {self.run_name}")
        print(f"Scenario ID: {self.scenario_id}")
        print(f"Source table: {self.source_table}")
        print(f"Iterations: {self.iterations}")
        print(f"Join strategy: {self.join_strategy}")
        print(f"Aggregation: {self.aggregation}")
        print(f"DL Variant: {self.target_dl}")

    def write_to_log(self, out_path):
        if Path(out_path).parent.exists():
            with open(out_path, "a") as fp:
                fp.write(self.to_string() + "\n")

    def write_to_json(self, root_path="results/json"):
        res_dict = copy.deepcopy(vars(self))
        results = self.results.clone()
        res_dict["results"] = results.to_dicts()
        res_dict["timestamps"] = {
            k: v.isoformat() for k, v in res_dict["timestamps"].items()
        }
        if Path(root_path).exists():
            with open(
                Path(root_path, self.run_name, f"{self.scenario_id}.json"), "w"
            ) as fp:
                json.dump(res_dict, fp, indent=2)
        else:
            raise IOError(f"Invalid path {root_path}")


class RunLogger:
    def __init__(
        self,
        scenario_logger: ScenarioLogger,
        fold_id: int,
        additional_parameters: dict,
        json_path="results/json",
    ):
        # TODO: rewrite with __getitem__ instead
        self.scenario_id = scenario_logger.scenario_id
        self.path_run_logs = scenario_logger.path_run_logs
        self.path_candidate_logs = scenario_logger.path_candidate_logs
        self.fold_id = fold_id
        self.run_id = scenario_logger.get_next_run_id()
        self.status = None
        self.timestamps = {}
        self.durations = {}
        self.parameters = self.get_parameters(scenario_logger, additional_parameters)
        self.results = {}
        self.json_path = json_path

        self.mark_time("run")

    def get_parameters(self, scenario_logger: ScenarioLogger, additional_parameters):
        parameters = {
            "source_table": scenario_logger.source_table,
            "candidate_table": "",
            "left_on": "",
            "right_on": "",
            "git_hash": scenario_logger.git_hash,
            "index_name": "base_table",
            "iterations": scenario_logger.iterations,
            "join_strategy": scenario_logger.join_strategy,
            "aggregation": scenario_logger.aggregation,
            "target_dl": scenario_logger.target_dl,
            "model_selection": scenario_logger.model_selection,
            "feature_selection": scenario_logger.feature_selection,
        }
        if additional_parameters is not None:
            parameters.update(additional_parameters)

        return parameters

    def update_timestamps(self, additional_timestamps=None):
        if additional_timestamps is not None:
            self.timestamps.update(additional_timestamps)

    def set_run_status(self, status):
        """Set run status for logging.

        Args:
            status (str): Status to use.
        """
        self.status = status

    def start_time(self, label, cumulative=False):
        """Wrapper around the `mark_time` function for better clarity.

        Args:
            label (str): Label of the operation to mark.
            cumulative (bool, optional): If set to true, all operations performed with the same label
            will add up to a total duration rather than being marked independently. Defaults to False.
        """
        return self.mark_time(label, cumulative)

    def end_time(self, label, cumulative=False):
        if label not in self.timestamps:
            raise KeyError(f"Label {label} was not found.")
        return self.mark_time(label, cumulative)

    def mark_time(self, label, cumulative=False):
        """Given a `label`, add a new timestamp if `label` isn't found, otherwise
        mark the end of the timestamp and add a new duration.

        Args:
            label (str): Label of the operation to mark.
            cumulative (bool, optional): If set to true, all operations performed with the same label
            will add up to a total duration rather than being marked independently. Defaults to False.

        """
        if label not in self.timestamps:
            self.timestamps[label] = [dt.datetime.now(), None]
            self.durations["time_" + label] = 0
        else:
            self.timestamps[label][1] = dt.datetime.now()
            this_segment = self.timestamps[label]
            if cumulative:
                self.durations["time_" + label] += (
                    this_segment[1] - this_segment[0]
                ).total_seconds()
            else:
                self.durations["time_" + label] = (
                    this_segment[1] - this_segment[0]
                ).total_seconds()

    def get_time(self, label):
        """Retrieve a time according to the given label.

        Args:
            label (str): Label of the timestamp to be retrieved.
        Returns:
            _type_: Retrieved timestamp.
        """
        if label in self.timestamps:
            return self.timestamps[label]
        else:
            raise KeyError(f"Label {label} not found in timestamps.")

    def get_duration(self, label):
        """Retrieve a duration according to the given label.

        Args:
            label (str): Label of the duration to be retrieved.
        Returns:
            _type_: Retrieved duration.

        Raises:
            KeyError if the provided label is not found.
        """
        if label in self.durations:
            return self.durations[label]
        else:
            raise KeyError(f"Label {label} not found in durations.")

    def to_str(self):
        res_str = ",".join(
            map(
                str,
                [
                    self.scenario_id,
                    self.run_id,
                    self.status,
                    self.parameters["target_dl"],
                    self.parameters["git_hash"],
                    self.parameters["index_name"],
                    self.parameters["source_table"],
                    self.parameters["candidate_table"],
                    self.parameters["iterations"],
                    self.parameters["join_strategy"],
                    self.parameters["aggregation"],
                    self.parameters["feature_selection"],
                    self.parameters["model_selection"],
                    self.fold_id,
                    self.durations.get("time_train", ""),
                    self.durations.get("time_eval", ""),
                    self.durations.get("time_join", ""),
                    self.durations.get("time_eval_join", ""),
                    self.results.get("n_cols", ""),
                    self.results.get("rmse", ""),
                    self.results.get("r2score", ""),
                ],
            )
        )
        return res_str

    def to_candidate_log_file(self):
        self.to_logfile(self.path_candidate_logs)

    def to_run_log_file(self):
        self.to_logfile(self.path_run_logs)

    def to_logfile(self, path_logfile):
        if Path(path_logfile).exists():
            with open(path_logfile, "a") as fp:
                fp.write(self.to_str() + "\n")
        else:
            with open(path_logfile, "w") as fp:
                col_list = [
                    "scenario_id",
                    "run_id",
                    "status",
                    "yadl_version",
                    "git_hash",
                    "index_name",
                    "base_table",
                    "candidate_table",
                    "iterations",
                    "join_strategy",
                    "aggregation",
                    "feature_selection",
                    "model_selection",
                    "fold_id",
                    "time_train",
                    "time_eval",
                    "time_join",
                    "time_eval_join",
                    "n_cols",
                    "rmse",
                    "r2score",
                ]
                header = ",".join(col_list)
                fp.write(header + "\n")
                fp.write(self.to_str() + "\n")

    def to_json(self):
        raise NotImplementedError
