import os
import pickle
from pathlib import Path

import polars as pl
from joblib import dump, load

from src.data_structures.indices import LazoIndex, MinHashIndex
from src.data_structures.metadata import (
    CandidateJoin,
    MetadataIndex,
    QueryResult,
    RawDataset,
)

DEFAULT_INDEX_DIR = Path("data/metadata/_indices")
DEFAULT_QUERY_RESULT_DIR = Path("results/generated_candidates")


def prepare_default_configs(data_dir, selected_indices=None):
    """Prepare default configurations for various indexing methods and provide the
    data directory that contains the metadata of the tables to be indexed.

    Args:
        data_dir (str): Path to the directory that contains the metadata.
        selected_indices (str, optional): If provided, prepare and run only the selected indices.

    Raises:
        IOError: Raise IOError if `data_dir` is incorrect.

    Returns:
        dict: Configuration dictionary
    """
    if Path(data_dir).exists():
        configs = {
            "lazo": {
                "data_dir": data_dir,
                "partition_size": 50_000,
                "host": "localhost",
                "port": 15449,
            },
            "minhash": {
                "data_dir": data_dir,
                "thresholds": [20],
                "oneshot": True,
                "num_perm": 128,
                "n_jobs": -1,
            },
        }
        if selected_indices is not None:
            return {
                index_name: config
                for index_name, config in configs.items()
                if index_name in selected_indices
            }
        else:
            return configs
    else:
        raise IOError(f"Invalid path {data_dir}")


def get_candidates(query_table, query_column, indices):
    """Given query table and column, query the required indices and produce the
    candidates. Used for debugging.

    Args:
        query_table (_type_): _description_
        query_column (_type_): _description_
        indices (_type_): _description_
    """
    pass


def write_candidates_on_file(candidates, output_file_path, separator=","):
    with open(output_file_path, "w") as fp:
        fp.write("tbl_pth1,tbl1,col1,tbl_pth2,tbl2,col2\n")

        for key, cand in candidates.items():
            rstr = cand.get_joinpath_str(sep=separator)
            fp.write(rstr + "\n")

    # open output file

    # write the candidates

    # metam format is left_table;left_on_column;right_table;right_on_column


def generate_candidates(
    index_name: str,
    index_result: list,
    metadata_index: MetadataIndex,
    mdata_source: dict,
    query_column: str,
    top_k=15,
):
    candidates = {}
    for res in index_result:
        hash_, column, similarity = res
        mdata_cand = metadata_index.query_by_hash(hash_)
        cjoin = CandidateJoin(
            indexing_method=index_name,
            source_table_metadata=mdata_source,
            candidate_table_metadata=mdata_cand,
            how="left",
            left_on=query_column,
            right_on=column,
            similarity_score=similarity,
        )
        candidates[cjoin.candidate_id] = cjoin

    if top_k > 0:
        # TODO rewrite this so it's cleaner
        ranking = [(k, v.similarity_score) for k, v in candidates.items()]
        clamped = [x[0] for x in sorted(ranking, key=lambda x: x[1], reverse=True)][
            :top_k
        ]

        candidates = {k: v for k, v in candidates.items() if k in clamped}
    return candidates


def load_index(data_lake_version, index_name):
    index_path = Path(
        DEFAULT_INDEX_DIR, data_lake_version, f"{index_name}_index.pickle"
    )
    if index_name == "minhash":
        with open(index_path, "rb") as fp:
            input_dict = load(fp)
        index = MinHashIndex()
        index.load_index(index_dict=input_dict)
    elif index_name == "lazo":
        index = LazoIndex()
        index.load_index(index_path)
    else:
        raise ValueError(f"Unknown index {index_name}.")
    return index


def query_index(
    index: MinHashIndex | LazoIndex, query_tab_path, query_column, mdata_index
):
    query_tab_metadata = RawDataset(
        query_tab_path.resolve(), "queries", "data/metadata/queries"
    )
    query_tab_metadata.save_metadata_to_json()

    query_result = QueryResult(index, query_tab_metadata, query_column, mdata_index)
    query_result.save_to_pickle()


def load_query_result(yadl_version, index_name, tab_name, query_column):
    query_result_path = "{}__{}__{}__{}.pickle".format(
        yadl_version,
        index_name,
        tab_name,
        query_column,
    )
    with open(Path(DEFAULT_QUERY_RESULT_DIR, query_result_path), "rb") as fp:
        query_result = pickle.load(fp)
    return query_result
