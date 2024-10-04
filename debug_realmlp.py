# %%
import datetime as dt
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from skrub import (
    AggJoiner,
    GapEncoder,
    MinHashEncoder,
    MultiAggJoiner,
    TableVectorizer,
    tabular_learner,
)

from src.utils.indexing import load_query_result

CACHE_PATH = "results/cache"


# %%
def prepare_skmodel():
    inner_model = make_pipeline(
        TableVectorizer(
            high_cardinality=MinHashEncoder(),
            low_cardinality=OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=np.nan
            ),
            n_jobs=32,
        ),
        HistGradientBoostingRegressor(),
        # RandomForestRegressor(n_jobs=32),
        memory=CACHE_PATH,
    )

    return inner_model


def fit_predict_skmodel(X_train, X_valid, y_train, model):

    model.fit(X_train, y_train)

    return model.predict(X_valid)


def prepare_catboost(X):
    defaults = {
        "l2_leaf_reg": 0.01,
        "od_type": "Iter",
        "od_wait": 10,
        "iterations": 100,
        "verbose": 0,
    }

    cat_features = X.select(cs.string()).columns

    parameters = dict(defaults)
    parameters["random_seed"] = 42
    return CatBoostRegressor(cat_features=cat_features, **parameters)


def prepare_table_catboost(table):
    table = table.fill_null(value="null").fill_nan(value=np.nan)
    return table.to_pandas()


def fit_predict_catboost(X_train, X_valid, y_train, model: CatBoostRegressor):
    model.fit(X_train, y_train)

    return model.predict(X_valid)


# %%
def prepare():
    df = pl.read_parquet(
        "data/source_tables/yadl/company_employees-yadl-depleted.parquet"
    )

    X = df.drop("target")
    y = df["target"]

    query_info = {
        "data_lake": "wordnet_full",
        "table_path": "data/source_tables/yadl/company_employees-yadl-depleted.parquet",
        "query_column": "col_to_embed",
        "top_k": 10,
        "join_discovery_method": "exact_matching",
    }

    query_tab_path = Path(query_info["table_path"])
    if not query_tab_path.exists():
        raise FileNotFoundError(f"File {query_tab_path} not found.")

    tab_name = query_tab_path.stem
    query_result = load_query_result(
        query_info["data_lake"],
        query_info["join_discovery_method"],
        tab_name,
        query_info["query_column"],
        top_k=query_info["top_k"],
    )

    candidate_joins = query_result.candidates

    return X, y, candidate_joins


#%%
def prep_cjoin(candidates):
    clean_cjoin = {}

    with open("candidates.txt", "w") as fp:
        for k, v in candidates.items():
            clean_cjoin[k] = {
                "candidate_path": Path(v.candidate_metadata["full_path"]).name,
                "left_on": v.left_on,
                "right_on": v.right_on,
            }
            fp.write(clean_cjoin[-k]["candidate_path"] + "\n")
    return clean_cjoin


#%%
X, y, cjoin = prepare()
#%%
clean_cjoin = prep_cjoin(cjoin)

pickle.dump(clean_cjoin, open("candidates.pickle", "wb"))
# %%
