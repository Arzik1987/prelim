import importlib.util
import io
import sys
import tarfile
import types
import zipfile
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"


def _load_get_data_module():
    module_name = "get_data_test_runner"
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, EXPERIMENTS_DIR / "get_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_has_no_side_effects(monkeypatch):
    calls = []
    monkeypatch.setattr("urllib.request.urlretrieve", lambda *args, **kwargs: calls.append((args, kwargs)))

    module = _load_get_data_module()

    assert calls == []
    assert callable(module.get_single)


def test_get_single_rejects_unknown_dataset():
    module = _load_get_data_module()

    with pytest.raises(ValueError):
        module.get_single("unknown-dataset")


def test_plain_download_creates_dataset_directory(monkeypatch, tmp_path):
    module = _load_get_data_module()
    recorded = []

    def fake_download(url, output_path, dname):
        output_path = Path(output_path)
        output_path.write_text("payload", encoding="utf-8")
        recorded.append((url, output_path, dname))

    monkeypatch.setattr(module, "download_url", fake_download)

    module.get_single("clean2", data_dir=tmp_path)

    assert recorded[0][2] == "clean2"
    assert recorded[0][1] == tmp_path / "clean2" / "clean2.tsv.gz"
    assert (tmp_path / "clean2").is_dir()


def test_zip_dataset_extracts_and_cleans_archive(monkeypatch, tmp_path):
    module = _load_get_data_module()

    def fake_download(url, output_path, dname):
        output_path = Path(output_path)
        with zipfile.ZipFile(output_path, "w") as zip_ref:
            zip_ref.writestr("sample.txt", "ok")

    monkeypatch.setattr(module, "download_url", fake_download)

    module.get_single("anuran", data_dir=tmp_path)

    assert not (tmp_path / "anuran.zip").exists()
    assert (tmp_path / "anuran" / "sample.txt").read_text(encoding="utf-8") == "ok"


def test_tar_dataset_renames_extracted_directory(monkeypatch, tmp_path):
    module = _load_get_data_module()

    def fake_download(url, output_path, dname):
        output_path = Path(output_path)
        with tarfile.open(output_path, "w:gz") as tar_ref:
            payload = b"content"
            info = tarfile.TarInfo(name="ml-prove/item.txt")
            info.size = len(payload)
            tar_ref.addfile(info, io.BytesIO(payload))

    monkeypatch.setattr(module, "download_url", fake_download)

    module.get_single("ml", data_dir=tmp_path)

    assert not (tmp_path / "ml.tar.gz").exists()
    assert (tmp_path / "ml" / "item.txt").read_text(encoding="utf-8") == "content"
    assert not (tmp_path / "ml-prove").exists()


def test_converter_path_writes_csv(monkeypatch, tmp_path):
    module = _load_get_data_module()

    def fake_download(url, output_path, dname):
        output_path = Path(output_path)
        with zipfile.ZipFile(output_path, "w") as zip_ref:
            zip_ref.writestr("3year.arff", "@RELATION demo")

    monkeypatch.setattr(module, "download_url", fake_download)
    monkeypatch.setattr(module, "arff", types.SimpleNamespace(load=lambda handle: {"data": [[1, 2], [3, 4]]}))

    module.get_single("bankruptcy", data_dir=tmp_path)

    result = pd.read_csv(tmp_path / "bankruptcy" / "3year.csv", header=None)
    assert result.values.tolist() == [[1, 2], [3, 4]]
