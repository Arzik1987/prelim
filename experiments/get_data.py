import argparse
import os
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.total = None
            self.n = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, n):
            self.n += n

try:
    import arff
except ModuleNotFoundError:
    arff = None


DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_DATASETS = (
    "anuran",
    "avila",
    "bankruptcy",
    "ccpp",
    "cc",
    "clean2",
    "dry",
    "ees",
    "electricity",
    "gas",
    "gt",
    "higgs",
    "htru",
    "jm1",
    "ml",
    "nomao",
    "occupancy",
    "parkinson",
    "pendata",
    "ring",
    "saac2",
    "seizure",
    "sensorless",
    "seoul",
    "shuttle",
    "stocks",
    "sylva",
    "turbine",
    "wine",
)


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path, dname):
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=dname) as progress:
        urllib.request.urlretrieve(url, filename=str(output_path), reporthook=progress.update_to)


def require_arff():
    if arff is None:
        raise ModuleNotFoundError("The 'arff' package is required for ARFF dataset conversion.")
    return arff


def convert_bankruptcy(data_dir, dataset_name):
    dataset_dir = data_dir / dataset_name
    frame = pd.DataFrame(require_arff().load(open(dataset_dir / "3year.arff"))["data"])
    frame.to_csv(dataset_dir / "3year.csv", index=False, header=False)


def convert_cc(data_dir, dataset_name):
    dataset_dir = data_dir / dataset_name
    frame = pd.read_excel(dataset_dir / "default of credit card clients.xls", header=1)
    frame.to_csv(dataset_dir / "default_of_credit_card_clients.csv", index=False)


def convert_ccpp(data_dir, dataset_name):
    dataset_dir = data_dir / dataset_name
    frame = pd.read_excel(dataset_dir / "Folds5x2_pp.ods", engine="odf")
    frame.to_csv(dataset_dir / "Folds5x2_pp.csv", index=False)


def convert_dry(data_dir, dataset_name):
    dataset_dir = data_dir / dataset_name
    frame = pd.DataFrame(require_arff().load(open(dataset_dir / "Dry_Bean_Dataset.arff"))["data"])
    frame.columns = [
        "Area",
        "Perimeter",
        "MajorAxisLength",
        "MinorAxisLength",
        "AspectRation",
        "Eccentricity",
        "ConvexArea",
        "EquivDiameter",
        "Extent",
        "Solidity",
        "roundness",
        "Compactness",
        "ShapeFactor1",
        "ShapeFactor2",
        "ShapeFactor3",
        "ShapeFactor4",
        "Class",
    ]
    frame.to_csv(dataset_dir / "Dry_Bean_Dataset.csv", index=False)


@dataclass(frozen=True)
class DatasetSpec:
    url: str
    download_relpath: str
    extract_relpath: str | None = None
    rename_pairs: tuple[tuple[str, str], ...] = ()
    converter: object = None
    cleanup_download: bool = False


DATASET_SPECS = {
    "anuran": DatasetSpec(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00406/Anuran%20Calls%20(MFCCs).zip",
        download_relpath="anuran.zip",
        extract_relpath="anuran",
        cleanup_download=True,
    ),
    "avila": DatasetSpec(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00459/avila.zip",
        download_relpath="avila.zip",
        extract_relpath=".",
        cleanup_download=True,
    ),
    "bankruptcy": DatasetSpec(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00365/data.zip",
        download_relpath="bankruptcy.zip",
        extract_relpath="bankruptcy",
        converter=convert_bankruptcy,
        cleanup_download=True,
    ),
    "cc": DatasetSpec(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls",
        download_relpath="cc/default of credit card clients.xls",
        converter=convert_cc,
    ),
    "ccpp": DatasetSpec(
        url="http://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip",
        download_relpath="ccpp.zip",
        extract_relpath=".",
        rename_pairs=(("CCPP", "ccpp"),),
        converter=convert_ccpp,
        cleanup_download=True,
    ),
    "clean2": DatasetSpec(
        url="https://github.com/EpistasisLab/pmlb/raw/master/datasets/clean2/clean2.tsv.gz",
        download_relpath="clean2/clean2.tsv.gz",
    ),
    "dry": DatasetSpec(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00602/DryBeanDataset.zip",
        download_relpath="dry.zip",
        extract_relpath=".",
        rename_pairs=(("DryBeanDataset", "dry"),),
        converter=convert_dry,
        cleanup_download=True,
    ),
    "ees": DatasetSpec(
        url="https://www.openml.org/data/get_csv/1587924/phplE7q6h",
        download_relpath="ees/phplE7q6h.csv",
    ),
    "electricity": DatasetSpec(
        url="https://www.openml.org/data/get_csv/2419/electricity-normalized.arff",
        download_relpath="electricity/electricity-normalized.csv",
    ),
    "gas": DatasetSpec(
        url="https://www.openml.org/data/get_csv/1588715/phpbL6t4U",
        download_relpath="gas/phpbL6t4U.csv",
    ),
    "gt": DatasetSpec(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data",
        download_relpath="gt/magic04.data",
    ),
    "higgs": DatasetSpec(
        url="https://www.openml.org/data/get_csv/2063675/phpZLgL9q",
        download_relpath="higgs/phpZLgL9q.csv",
    ),
    "htru": DatasetSpec(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00372/HTRU2.zip",
        download_relpath="htru.zip",
        extract_relpath="htru",
        cleanup_download=True,
    ),
    "jm1": DatasetSpec(
        url="https://www.openml.org/data/get_csv/53936/jm1.arff",
        download_relpath="jm1/jm1.csv",
    ),
    "ml": DatasetSpec(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00249/ml-prove.tar.gz",
        download_relpath="ml.tar.gz",
        extract_relpath=".",
        rename_pairs=(("ml-prove", "ml"),),
        cleanup_download=True,
    ),
    "nomao": DatasetSpec(
        url="https://www.openml.org/data/get_csv/1592278/phpDYCOet",
        download_relpath="nomao/phpDYCOet.csv",
    ),
    "occupancy": DatasetSpec(
        url="http://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip",
        download_relpath="occupancy.zip",
        extract_relpath="occupancy",
        cleanup_download=True,
    ),
    "parkinson": DatasetSpec(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data",
        download_relpath="parkinson/parkinsons_updrs.data",
    ),
    "pendata": DatasetSpec(
        url="https://www.openml.org/data/get_csv/32/dataset_32_pendigits.arff",
        download_relpath="pendata/dataset_32_pendigits.csv",
    ),
    "ring": DatasetSpec(
        url="https://github.com/EpistasisLab/pmlb/raw/master/datasets/ring/ring.tsv.gz",
        download_relpath="ring/ring.tsv.gz",
    ),
    "saac2": DatasetSpec(
        url="https://www.openml.org/data/get_csv/21230748/SAAC2.arff",
        download_relpath="saac2/SAAC2.csv",
    ),
    "seizure": DatasetSpec(
        url="https://github.com/akshayg056/Epileptic-seizure-detection-/raw/master/data.csv",
        download_relpath="seizure/data.csv",
    ),
    "sensorless": DatasetSpec(
        url="http://archive.ics.uci.edu/ml/machine-learning-databases/00325/Sensorless_drive_diagnosis.txt",
        download_relpath="sensorless/Sensorless_drive_diagnosis.txt",
    ),
    "seoul": DatasetSpec(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv",
        download_relpath="seoul/SeoulBikeData.csv",
    ),
    "shuttle": DatasetSpec(
        url="https://github.com/EpistasisLab/pmlb/raw/master/datasets/shuttle/shuttle.tsv.gz",
        download_relpath="shuttle/shuttle.tsv.gz",
    ),
    "stocks": DatasetSpec(
        url="https://www.openml.org/data/get_csv/2160285/phpg2t68G",
        download_relpath="stocks/phpg2t68G.csv",
    ),
    "sylva": DatasetSpec(
        url="https://www.openml.org/data/get_csv/53923/sylva_prior.arff",
        download_relpath="sylva/sylva_prior.csv",
    ),
    "turbine": DatasetSpec(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00551/pp_gas_emission.zip",
        download_relpath="turbine.zip",
        extract_relpath="turbine",
        cleanup_download=True,
    ),
    "wine": DatasetSpec(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        download_relpath="wine/winequality-red.csv",
    ),
}


EXTRA_DOWNLOADS = {
    "wine": (
        (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
            "wine/winequality-white.csv",
        ),
    ),
}


def ensure_parent(path):
    path.parent.mkdir(parents=True, exist_ok=True)


def extract_zip(archive_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)


def extract_tar_gz(archive_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar_ref:
        tar_ref.extractall(output_dir)


def apply_renames(data_dir, rename_pairs):
    for source_relpath, target_relpath in rename_pairs:
        (data_dir / source_relpath).rename(data_dir / target_relpath)


def materialize_dataset(dataset_name, spec, data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    download_path = data_dir / spec.download_relpath
    ensure_parent(download_path)
    download_url(spec.url, download_path, dataset_name)

    if spec.extract_relpath is not None:
        extract_dir = data_dir / spec.extract_relpath
        if download_path.suffix == ".zip":
            extract_zip(download_path, extract_dir)
        elif download_path.name.endswith(".tar.gz"):
            extract_tar_gz(download_path, extract_dir)
        else:
            raise ValueError("Unsupported archive type for %s" % download_path.name)
        if spec.cleanup_download:
            download_path.unlink()

    if spec.rename_pairs:
        apply_renames(data_dir, spec.rename_pairs)

    if spec.converter is not None:
        print("converting to csv ...")
        spec.converter(data_dir, dataset_name)

    for extra_url, extra_relpath in EXTRA_DOWNLOADS.get(dataset_name, ()):
        extra_path = data_dir / extra_relpath
        ensure_parent(extra_path)
        download_url(extra_url, extra_path, dataset_name)


def get_single(dname, data_dir=DATA_DIR):
    if dname not in DATASET_SPECS:
        raise ValueError("%r is a wrong dataset name" % dname)
    materialize_dataset(dname, DATASET_SPECS[dname], Path(data_dir))


def get_multiple(dnames, data_dir=DATA_DIR):
    for dname in dnames:
        get_single(dname, data_dir=data_dir)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Download experiment datasets.")
    parser.add_argument(
        "datasets",
        nargs="*",
        default=list(DEFAULT_DATASETS),
        help="Dataset names to download. Defaults to the full experiment set.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help="Target directory for downloaded datasets.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    get_multiple(args.datasets, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
