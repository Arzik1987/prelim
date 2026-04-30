from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def nunique(a, axis):
    return (np.diff(np.sort(a, axis=axis), axis=axis) != 0).sum(axis=axis) + 1


def _read_csv(data_dir, relpath, **kwargs):
    return pd.read_csv(Path(data_dir) / relpath, **kwargs)


def _read_many_csv(data_dir, relpaths, **kwargs):
    return pd.concat([_read_csv(data_dir, relpath, **kwargs) for relpath in relpaths], ignore_index=True)


def _select_columns(frame, columns):
    if callable(columns):
        selected = columns(frame)
        if isinstance(selected, pd.DataFrame):
            return selected.to_numpy()
        if isinstance(selected, (pd.Series, pd.Index)):
            return frame[selected].to_numpy()
        return frame.loc[:, selected].to_numpy()
    return frame.loc[:, columns].to_numpy()


def _build_xy(frame, target, positive, feature_columns, cast_float=False):
    y = (frame[target] == positive).astype(int).to_numpy()
    X = _select_columns(frame, feature_columns)
    if cast_float:
        X = X.copy()
        X[X == "?"] = "nan"
        X = X.astype(np.float64)
    return X, y


def _drop_nan_rows(X, y):
    mask = ~pd.isna(X).any(axis=1)
    return X[mask], y[mask]


def _load_occupancy(data_dir):
    frame = _read_many_csv(
        data_dir,
        (
            "occupancy/datatest.txt",
            "occupancy/datatest2.txt",
            "occupancy/datatraining.txt",
        ),
        delimiter=",",
    )
    frame["date"] = pd.to_datetime(frame["date"]).dt.hour
    return _build_xy(frame, "Occupancy", 1, lambda df: df[df.columns.drop("Occupancy")])


def _load_higgs7(data_dir):
    frame = _read_csv(data_dir, "higgs/phpZLgL9q.csv", delimiter=",", nrows=98049)
    return _build_xy(frame, "class", 1, lambda df: df.filter(regex="m_", axis=1))


def _load_electricity(data_dir):
    frame = _read_csv(data_dir, "electricity/electricity-normalized.csv", delimiter=",")
    return _build_xy(frame, "class", "UP", lambda df: df[df.columns.drop("class")])


def _load_htru(data_dir):
    frame = _read_csv(data_dir, "htru/HTRU_2.csv", delimiter=",", header=None)
    return _build_xy(frame, 8, 1, lambda df: df.iloc[:, :8])


def _load_shuttle(data_dir):
    frame = _read_csv(data_dir, "shuttle/shuttle.tsv.gz", delimiter="\t")
    return _build_xy(frame, "target", 1, lambda df: df[df.columns.drop("target")])


def _load_avila(data_dir):
    frame = _read_many_csv(
        data_dir,
        ("avila/avila-tr.txt", "avila/avila-ts.txt"),
        delimiter=",",
        header=None,
    )
    return _build_xy(frame, 10, "A", lambda df: df.iloc[:, :10])


def _load_gt(data_dir):
    frame = _read_csv(data_dir, "gt/magic04.data", delimiter=",", header=None)
    return _build_xy(frame, 10, "g", lambda df: df.iloc[:, :10])


def _load_cc(data_dir):
    frame = _read_csv(data_dir, "cc/default_of_credit_card_clients.csv", delimiter=",")
    return _build_xy(frame, frame.columns[24], 1, lambda df: df.iloc[:, 1:24])


def _load_ees(data_dir):
    frame = _read_csv(data_dir, "ees/phplE7q6h.csv", delimiter=",")
    return _build_xy(frame, "Class", 1, lambda df: df.iloc[:, :14])


def _load_pendata(data_dir):
    frame = _read_csv(data_dir, "pendata/dataset_32_pendigits.csv", delimiter=",")
    return _build_xy(frame, "class", 1, lambda df: df[df.columns.drop("class")])


def _load_ring(data_dir):
    frame = _read_csv(data_dir, "ring/ring.tsv.gz", delimiter="\t")
    return _build_xy(frame, "target", 1, lambda df: df[df.columns.drop("target")])


def _load_sylva(data_dir):
    frame = _read_csv(data_dir, "sylva/sylva_prior.csv", delimiter=",")
    return _build_xy(frame, "label", 1, lambda df: df[df.columns.drop("label")])


def _load_higgs21(data_dir):
    frame = _read_csv(data_dir, "higgs/phpZLgL9q.csv", delimiter=",", nrows=98049)
    y = (frame["class"] == 1).astype(int).to_numpy()
    features = frame[frame.columns.drop("class")]
    X = features[features.columns.drop(list(features.filter(regex="m_", axis=1)))].to_numpy()
    return X, y


def _load_jm1(data_dir):
    frame = _read_csv(data_dir, "jm1/jm1.csv", delimiter=",")
    return _build_xy(frame, "defects", True, lambda df: df[df.columns.drop("defects")], cast_float=True)


def _load_saac2(data_dir):
    frame = _read_csv(data_dir, "saac2/SAAC2.csv", delimiter=",")
    return _build_xy(frame, "class", 1, lambda df: df[df.columns.drop("class")], cast_float=True)


def _load_stocks(data_dir):
    frame = _read_csv(data_dir, "stocks/phpg2t68G.csv", delimiter=",")
    return _build_xy(frame, "attribute_21", 1, lambda df: df[df.columns.drop("attribute_21")])


def _load_sensorless(data_dir):
    frame = _read_csv(data_dir, "sensorless/Sensorless_drive_diagnosis.txt", delimiter=" ", header=None)
    return _build_xy(frame, 48, 1, lambda df: df.iloc[:, :48])


def _load_bankruptcy(data_dir):
    frame = _read_csv(data_dir, "bankruptcy/3year.csv", delimiter=",", header=None)
    return _build_xy(frame, 64, 1, lambda df: df.iloc[:, :64])


def _load_nomao(data_dir):
    frame = _read_csv(data_dir, "nomao/phpDYCOet.csv", delimiter=",")
    return _build_xy(frame, "Class", 1, lambda df: df[df.columns.drop("Class")])


def _load_gas(data_dir):
    frame = _read_csv(data_dir, "gas/phpbL6t4U.csv", delimiter=",")
    return _build_xy(frame, "Class", 1, lambda df: df.iloc[:, :128])


def _load_clean2(data_dir):
    frame = _read_csv(data_dir, "clean2/clean2.tsv.gz", delimiter="\t")
    return _build_xy(frame, "target", 1, lambda df: df.filter(regex="^f", axis=1))


def _load_seizure(data_dir):
    frame = _read_csv(data_dir, "seizure/data.csv", delimiter=",")
    return _build_xy(frame, "y", 1, lambda df: df.filter(regex="X", axis=1))


def _load_smartphone(data_dir):
    frame = _read_csv(data_dir, "smartphone/php88ZB4Q.csv", delimiter=",")
    return _build_xy(frame, "Class", 1, lambda df: df[df.columns.drop("Class")])


def _load_ccpp(data_dir):
    frame = _read_csv(data_dir, "ccpp/Folds5x2_pp.csv", delimiter=",")
    y = (frame["PE"] > 455).astype(int).to_numpy()
    X = frame[frame.columns.drop("PE")].to_numpy()
    return X, y


def _load_seoul(data_dir):
    frame = _read_csv(data_dir, "seoul/SeoulBikeData.csv", delimiter=",", skiprows=1, header=None)
    y = (frame.iloc[:, 1] > 800).astype(int).to_numpy()
    X = frame.iloc[:, 2:11].to_numpy()
    return X, y


def _load_turbine(data_dir):
    frame = _read_many_csv(
        data_dir,
        (
            "turbine/gt_2011.csv",
            "turbine/gt_2012.csv",
            "turbine/gt_2013.csv",
            "turbine/gt_2014.csv",
            "turbine/gt_2015.csv",
        ),
        delimiter=",",
    )
    y = (frame["NOX"] > 70).astype(int).to_numpy()
    X = frame.iloc[:, :9].to_numpy()
    return X, y


def _load_wine(data_dir):
    frame = _read_csv(data_dir, "wine/winequality-white.csv", delimiter=";")
    return _build_xy(frame, "quality", 6, lambda df: df[df.columns.drop("quality")])


def _load_parkinson(data_dir):
    frame = _read_csv(data_dir, "parkinson/parkinsons_updrs.data", delimiter=",")
    y = (frame["motor_UPDRS"] > 23).astype(int).to_numpy()
    X = frame.iloc[:, 6:].to_numpy()
    return X, y


def _load_dry(data_dir):
    frame = _read_csv(data_dir, "dry/Dry_Bean_Dataset.csv", delimiter=",")
    return _build_xy(frame, "Class", "DERMASON", lambda df: df[df.columns.drop("Class")])


def _load_anuran(data_dir):
    frame = _read_csv(data_dir, "anuran/Frogs_MFCCs.csv", delimiter=",")
    y = (frame["Family"] == "Hylidae").astype(int).to_numpy()
    X = frame.iloc[:, :22].to_numpy()
    return X, y


def _load_ml(data_dir):
    frame = _read_many_csv(
        data_dir,
        ("ml/train.csv", "ml/test.csv", "ml/validation.csv"),
        delimiter=",",
        header=None,
    )
    return _build_xy(frame, 56, 1, lambda df: df.iloc[:, :51])


@dataclass(frozen=True)
class DatasetLoader:
    loader: callable


DATASET_LOADERS = {
    "occupancy": DatasetLoader(_load_occupancy),
    "higgs7": DatasetLoader(_load_higgs7),
    "electricity": DatasetLoader(_load_electricity),
    "htru": DatasetLoader(_load_htru),
    "shuttle": DatasetLoader(_load_shuttle),
    "avila": DatasetLoader(_load_avila),
    "gt": DatasetLoader(_load_gt),
    "cc": DatasetLoader(_load_cc),
    "ees": DatasetLoader(_load_ees),
    "pendata": DatasetLoader(_load_pendata),
    "ring": DatasetLoader(_load_ring),
    "sylva": DatasetLoader(_load_sylva),
    "higgs21": DatasetLoader(_load_higgs21),
    "jm1": DatasetLoader(_load_jm1),
    "saac2": DatasetLoader(_load_saac2),
    "stocks": DatasetLoader(_load_stocks),
    "sensorless": DatasetLoader(_load_sensorless),
    "bankruptcy": DatasetLoader(_load_bankruptcy),
    "nomao": DatasetLoader(_load_nomao),
    "gas": DatasetLoader(_load_gas),
    "clean2": DatasetLoader(_load_clean2),
    "seizure": DatasetLoader(_load_seizure),
    "smartphone": DatasetLoader(_load_smartphone),
    "ccpp": DatasetLoader(_load_ccpp),
    "seoul": DatasetLoader(_load_seoul),
    "turbine": DatasetLoader(_load_turbine),
    "wine": DatasetLoader(_load_wine),
    "parkinson": DatasetLoader(_load_parkinson),
    "dry": DatasetLoader(_load_dry),
    "anuran": DatasetLoader(_load_anuran),
    "ml": DatasetLoader(_load_ml),
}


def load_data(dname, data_dir=DATA_DIR):
    if dname not in DATASET_LOADERS:
        raise ValueError("Unknown dataset name: %s" % dname)

    X, y = DATASET_LOADERS[dname].loader(Path(data_dir))
    return _drop_nan_rows(X, y)
