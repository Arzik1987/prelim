import os
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
REGISTRY_DIR = os.path.join(EXPERIMENTS_DIR, 'registry')
RUNS_DIR = os.path.join(REGISTRY_DIR, 'runs')
LATEST_RUN_PATH = os.path.join(REGISTRY_DIR, 'latest_run.txt')
LAYOUT_VERSION = 2

DEFAULT_DATASET_NAMES = (
    'clean2', 'seizure', 'gas', 'nomao', 'bankruptcy', 'anuran', 'avila',
    'ccpp', 'cc', 'dry', 'ees', 'electricity', 'gt', 'higgs21', 'higgs7',
    'htru', 'jm1', 'ml', 'occupancy', 'parkinson', 'pendata', 'ring',
    'saac2', 'sensorless', 'seoul', 'shuttle', 'stocks', 'sylva', 'turbine',
    'wine',
)
DEFAULT_DATASET_SIZES = (400, 100)
DEFAULT_VVA_GRID = (0.5, 1.0, 1.5, 2.0, 2.5)


def utc_timestamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def default_run_id():
    return 'run-' + utc_timestamp()


def parse_csv_list(raw_value, cast = str):
    values = []
    for item in raw_value.split(','):
        item = item.strip()
        if item:
            values.append(cast(item))
    return tuple(values)


def git_revision(repo_dir):
    try:
        head = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd = repo_dir,
            check = True,
            capture_output = True,
            text = True,
        ).stdout.strip()
        dirty = subprocess.run(
            ['git', 'status', '--short'],
            cwd = repo_dir,
            check = True,
            capture_output = True,
            text = True,
        ).stdout.strip()
        return {
            'commit': head,
            'dirty': bool(dirty),
        }
    except Exception:
        return {
            'commit': None,
            'dirty': None,
        }


@dataclass(frozen = True)
class ExperimentConfig:
    run_id: str
    datasets: tuple[str, ...]
    dataset_sizes: tuple[int, ...]
    nsets: int = 25
    split_seed: int = 2020
    generated_sample_size: int = 100000
    rules_sample_size: int = 10000
    ssl_pool_size: int = 10000
    vva_grid: tuple[float, ...] = field(default_factory = lambda: DEFAULT_VVA_GRID)
    jobs: int = field(default_factory = lambda: os.cpu_count() or 1)
    resume: bool = False
    registry_dir: str = REGISTRY_DIR

    @property
    def split_indices(self):
        return tuple(range(self.nsets))

    @property
    def runs_dir(self):
        return os.path.join(self.registry_dir, 'runs')

    @property
    def run_dir(self):
        return os.path.join(self.runs_dir, self.run_id)

    @property
    def raw_dir(self):
        return os.path.join(self.run_dir, 'raw')

    @property
    def derived_dir(self):
        return os.path.join(self.run_dir, 'derived')

    @property
    def figures_dir(self):
        return os.path.join(self.run_dir, 'figures')

    @property
    def manifest_path(self):
        return os.path.join(self.run_dir, 'manifest.json')

    @property
    def log_path(self):
        return os.path.join(self.run_dir, 'errors.log')

    def to_manifest(self):
        return {
            'layout_version': LAYOUT_VERSION,
            'run_id': self.run_id,
            'created_at_utc': utc_timestamp(),
            'host': socket.gethostname(),
            'platform': platform.platform(),
            'python_version': sys.version,
            'git': git_revision(EXPERIMENTS_DIR),
            'config': asdict(self),
        }


def ensure_registry_dir():
    os.makedirs(REGISTRY_DIR, exist_ok = True)
    os.makedirs(RUNS_DIR, exist_ok = True)


def write_latest_run(run_id, registry_dir = REGISTRY_DIR):
    os.makedirs(registry_dir, exist_ok = True)
    latest_run_path = os.path.join(registry_dir, 'latest_run.txt')
    with open(latest_run_path, 'w', encoding = 'utf-8') as handle:
        handle.write(run_id + '\n')


def read_latest_run_id(registry_dir = REGISTRY_DIR):
    latest_run_path = os.path.join(registry_dir, 'latest_run.txt')
    if not os.path.exists(latest_run_path):
        return None
    with open(latest_run_path, 'r', encoding = 'utf-8') as handle:
        run_id = handle.read().strip()
    return run_id or None


def ensure_run_layout(config):
    ensure_registry_dir()
    if os.path.exists(config.run_dir) and not config.resume:
        raise FileExistsError(
            'Run directory already exists: %s. Use --resume or choose a new --run-id.'
            % config.run_dir
        )

    os.makedirs(config.raw_dir, exist_ok = True)
    os.makedirs(config.derived_dir, exist_ok = True)
    os.makedirs(config.figures_dir, exist_ok = True)
    write_latest_run(config.run_id, config.registry_dir)


def resolve_run_dir(run_id):
    return os.path.join(RUNS_DIR, run_id)
