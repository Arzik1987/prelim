import logging
import os
import re
from pathlib import Path

import numpy as np

from .base import BaseGenerator


_RULE_RE = re.compile(r"^(?P<conditions>.*):\s*(?P<label>\S+)\s+\((?P<count>[0-9.]+)(?:/[0-9.]+)?\)")
_CONDITION_RE = re.compile(r"^x(?P<feature>\d+)\s*(?P<op><=|>=|<|>|=)\s*(?P<value>[-+0-9.eE]+)$")


class Gen_part(BaseGenerator):
    def __init__(self, seed=2020, min_samples=2, confidence=0.25, include_default=False):
        super().__init__("cmmpart", seed=seed)
        self.min_samples_ = min_samples
        self.confidence_ = confidence
        self.include_default_ = include_default
        self.boxes_ = None
        self.nsamples_ = None
        self.rules_ = None
        self.bounds_ = None

    def fit(self, X, y=None, metamodel=None):
        if y is None:
            raise ValueError("Gen_part.fit requires y")

        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        self.bounds_ = np.vstack((X.min(axis=0), X.max(axis=0)))
        output = self._fit_part_model(X, y)
        boxes, nsamples, rules = self._parse_part_rules(output, self.bounds_, self.include_default_)

        if len(boxes) == 0:
            boxes = [self.bounds_.copy()]
            nsamples = [len(X)]
            rules = ["<default>"]

        self.boxes_ = boxes
        self.nsamples_ = np.asarray(nsamples, dtype=int)
        self.rules_ = rules
        return self

    def _fit_part_model(self, X, y):
        try:
            import weka.core.jvm as jvm
            from weka.classifiers import Classifier
            from weka.core.dataset import create_instances_from_matrices
        except ImportError as exc:
            raise RuntimeError("Gen_part requires python-weka-wrapper3") from exc

        weka_home = Path(os.environ.get("WEKA_HOME", "/tmp/prelim-weka"))
        weka_home.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("WEKA_HOME", str(weka_home))

        if not jvm.started:
            jvm.start(packages=False, logging_level=logging.WARNING)

        cols_x = [f"x{i}" for i in range(X.shape[1])]
        data = create_instances_from_matrices(
            X,
            y,
            name="prelim_part",
            cols_x=cols_x,
            col_y="class",
            nominal_y=True,
        )
        data.class_is_last()

        classifier = Classifier(
            classname="weka.classifiers.rules.PART",
            options=[
                "-M",
                str(self.min_samples_),
                "-C",
                str(self.confidence_),
                "-Q",
                str(self.seed_),
            ],
        )
        classifier.build_classifier(data)
        return str(classifier)

    @staticmethod
    def _parse_part_rules(output, bounds, include_default=False):
        boxes = []
        nsamples = []
        rules = []

        for raw_line in output.splitlines():
            line = raw_line.strip()
            match = _RULE_RE.match(line)
            if match is None:
                continue

            conditions = match.group("conditions").strip()
            if not conditions and not include_default:
                continue

            box = bounds.copy()
            if conditions:
                for condition in conditions.split(" AND "):
                    cond_match = _CONDITION_RE.match(condition.strip())
                    if cond_match is None:
                        raise ValueError(f"Unsupported PART rule condition: {condition}")

                    feature = int(cond_match.group("feature"))
                    op = cond_match.group("op")
                    value = float(cond_match.group("value"))

                    if op in ("<=", "<"):
                        box[1, feature] = min(box[1, feature], value)
                    elif op in (">=", ">"):
                        box[0, feature] = max(box[0, feature], value)
                    else:
                        box[0, feature] = max(box[0, feature], value)
                        box[1, feature] = min(box[1, feature], value)

            if np.any(box[1, :] < box[0, :]):
                continue

            count = int(np.ceil(float(match.group("count"))))
            if count <= 0:
                continue

            boxes.append(box)
            nsamples.append(count)
            rules.append(line)

        return boxes, nsamples, rules

    def sample(self, n_samples=1):
        if self.boxes_ is None or self.nsamples_ is None:
            raise RuntimeError("Gen_part.sample called before fit")

        total = int(sum(self.nsamples_))
        if total <= 0:
            raise RuntimeError("Gen_part has no covered rule regions to sample from")

        niter = int(np.ceil(n_samples / total))
        X = []

        for _ in range(niter):
            for box, count in zip(self.boxes_, self.nsamples_):
                sidelen = box[1, :] - box[0, :]
                X.append(self.rng_.random_sample((count, len(sidelen))) * sidelen + box[0, :])

        X = np.concatenate(X)
        xdim = X.shape[0]
        X = X[self.rng_.choice(np.arange(xdim), size=xdim, replace=False), :].copy()
        return X[0:n_samples, :]
