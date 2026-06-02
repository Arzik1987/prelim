from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BaseGenerator


@dataclass
class _ColumnSpec:
    index: int
    is_numeric: bool
    categories: tuple | None = None
    mean: float | None = None
    scale: float | None = None
    dtype: np.dtype | None = None


class Gen_gibbs(BaseGenerator):
    def __init__(
        self,
        model_kwargs: dict | None = None,
        train_kwargs: dict | None = None,
        sample_kwargs: dict | None = None,
        seed=2020,
    ):
        super().__init__("gibbs", seed=seed)
        self.model_kwargs_ = {} if model_kwargs is None else dict(model_kwargs)
        self.train_kwargs_ = {} if train_kwargs is None else dict(train_kwargs)
        self.sample_kwargs_ = {} if sample_kwargs is None else dict(sample_kwargs)
        self.model_kwargs_.setdefault("hidden_dim", 64)
        self.model_kwargs_.setdefault("num_layers", 2)
        self.model_kwargs_.setdefault("num_heads", 4)
        self.model_kwargs_.setdefault("dropout", 0.1)
        self.model_kwargs_.setdefault("mixture_components", 5)
        self.train_kwargs_.setdefault("epochs", 50)
        self.train_kwargs_.setdefault("batch_size", 128)
        self.train_kwargs_.setdefault("lr", 3e-4)
        self.train_kwargs_.setdefault("weight_decay", 1e-6)
        self.train_kwargs_.setdefault("grad_clip", 5.0)
        self.train_kwargs_.setdefault("device", "cpu")
        self.sample_kwargs_.setdefault("gibbs_rounds", 3)
        self.sample_kwargs_.setdefault("batch_size", 512)
        self.X_ = None
        self.column_specs_ = None
        self.encoded_X_ = None
        self.estimator_ = None

    def fit(self, X, y=None, metamodel=None):
        del y, metamodel
        self.X_ = np.asarray(X, dtype=object).copy()
        if self.X_.ndim != 2:
            raise ValueError("X must be a 2D array")
        if len(self.X_) == 0:
            raise ValueError("X must contain at least one row")

        self.column_specs_ = self._analyze_columns(self.X_)
        self.encoded_X_ = self._encode_rows(self.X_)
        self.estimator_ = self._build_estimator()
        self.estimator_.fit(
            self.encoded_X_,
            self.column_specs_,
            train_kwargs=dict(self.train_kwargs_),
            sample_seed=self.seed_,
        )
        return self

    def sample(self, n_samples=1):
        if self.estimator_ is None or self.encoded_X_ is None or self.column_specs_ is None:
            raise RuntimeError("Generator must be fitted before sampling")
        if n_samples < 1:
            raise ValueError("n_samples must be positive")

        sampled = self.estimator_.sample(
            n_samples=n_samples,
            train_data=self.encoded_X_,
            column_specs=self.column_specs_,
            sample_kwargs=dict(self.sample_kwargs_),
            sample_seed=self.seed_,
        )
        sampled = np.asarray(sampled, dtype=float)
        if sampled.shape != (n_samples, self.encoded_X_.shape[1]):
            raise RuntimeError(
                f"Gibbs backend returned shape {sampled.shape}, expected {(n_samples, self.encoded_X_.shape[1])}"
            )
        return self._decode_rows(sampled)

    def _build_estimator(self):
        return _build_torch_estimator(self.model_kwargs_)

    def _analyze_columns(self, X):
        specs = []
        for index in range(X.shape[1]):
            column = X[:, index]
            dtype = np.asarray(column).dtype
            if self._is_numeric_column(column):
                numeric = np.asarray(column, dtype=float)
                scale = float(np.std(numeric))
                if scale <= 1e-8:
                    scale = 1.0
                specs.append(
                    _ColumnSpec(
                        index=index,
                        is_numeric=True,
                        mean=float(np.mean(numeric)),
                        scale=scale,
                        dtype=dtype,
                    )
                )
                continue

            categories = tuple(dict.fromkeys(column.tolist()))
            specs.append(
                _ColumnSpec(
                    index=index,
                    is_numeric=False,
                    categories=categories,
                    dtype=dtype,
                )
            )
        return specs

    def _encode_rows(self, X):
        encoded = np.zeros(X.shape, dtype=float)
        for spec in self.column_specs_:
            column = X[:, spec.index]
            if spec.is_numeric:
                encoded[:, spec.index] = (np.asarray(column, dtype=float) - spec.mean) / spec.scale
                continue

            category_to_index = {category: idx for idx, category in enumerate(spec.categories)}
            encoded[:, spec.index] = [category_to_index[value] for value in column]
        return encoded

    def _decode_rows(self, rows):
        decoded = np.empty(rows.shape, dtype=object)
        for spec in self.column_specs_:
            column = rows[:, spec.index]
            if spec.is_numeric:
                numeric = column * spec.scale + spec.mean
                cast_column = np.asarray(numeric, dtype=spec.dtype if spec.dtype is not None else float)
                decoded[:, spec.index] = cast_column
                continue

            indices = np.rint(column).astype(int)
            indices = np.clip(indices, 0, len(spec.categories) - 1)
            decoded[:, spec.index] = [spec.categories[idx] for idx in indices]
        if all(spec.is_numeric for spec in self.column_specs_):
            return np.asarray(decoded, dtype=float)
        return decoded

    @staticmethod
    def _is_numeric_column(column):
        try:
            np.asarray(column, dtype=float)
        except (TypeError, ValueError):
            return False
        return True


def _build_torch_estimator(model_kwargs):
    try:
        import torch
        import torch.nn.functional as F
        from torch import nn
    except Exception as exc:
        raise ImportError("Gen_gibbs requires PyTorch. Install torch to use this generator.") from exc

    mixture_components = int(model_kwargs["mixture_components"])
    hidden_dim = int(model_kwargs["hidden_dim"])
    num_layers = int(model_kwargs["num_layers"])
    num_heads = int(model_kwargs["num_heads"])
    dropout = float(model_kwargs["dropout"])

    class _MaskedTransformer(nn.Module):
        def __init__(self, column_specs):
            super().__init__()
            self.column_specs = column_specs
            self.hidden_dim = hidden_dim
            self.feature_embeddings = nn.Parameter(torch.randn(len(column_specs), hidden_dim) * 0.02)
            self.mask_embeddings = nn.Parameter(torch.randn(len(column_specs), hidden_dim) * 0.02)
            self.numeric_weight = nn.Parameter(torch.randn(len(column_specs), hidden_dim) * 0.02)
            self.numeric_bias = nn.Parameter(torch.zeros(len(column_specs), hidden_dim))
            self.categorical_embeddings = nn.ModuleList()
            self.numeric_heads = nn.ModuleList()
            self.categorical_heads = nn.ModuleList()
            for spec in column_specs:
                if spec.is_numeric:
                    self.categorical_embeddings.append(nn.Identity())
                    self.numeric_heads.append(nn.Linear(hidden_dim, mixture_components * 3))
                    self.categorical_heads.append(nn.Identity())
                else:
                    self.categorical_embeddings.append(nn.Embedding(len(spec.categories), hidden_dim))
                    self.numeric_heads.append(nn.Identity())
                    self.categorical_heads.append(nn.Linear(hidden_dim, len(spec.categories)))

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dropout=dropout,
                batch_first=True,
                dim_feedforward=hidden_dim * 4,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        def _embed(self, data):
            tokens = []
            for spec in self.column_specs:
                values = data[:, spec.index]
                if spec.is_numeric:
                    token = values.unsqueeze(-1) * self.numeric_weight[spec.index] + self.numeric_bias[spec.index]
                else:
                    token = self.categorical_embeddings[spec.index](values.long())
                token = token + self.feature_embeddings[spec.index]
                tokens.append(token)
            return torch.stack(tokens, dim=1)

        def forward(self, data, target_index):
            tokens = self._embed(data)
            tokens[:, target_index, :] = self.mask_embeddings[target_index] + self.feature_embeddings[target_index]
            encoded = self.encoder(tokens)
            return encoded[:, target_index, :]

        def loss(self, data, target_index):
            context = self.forward(data, target_index)
            spec = self.column_specs[target_index]
            target = data[:, target_index]
            if spec.is_numeric:
                params = self.numeric_heads[target_index](context)
                logits, means, log_scales = torch.chunk(params, 3, dim=-1)
                log_scales = torch.clamp(log_scales, min=-7.0, max=5.0)
                target = target.unsqueeze(-1)
                log_weights = F.log_softmax(logits, dim=-1)
                sq_term = -0.5 * ((target - means) / torch.exp(log_scales)) ** 2
                log_probs = log_weights + sq_term - log_scales - 0.5 * np.log(2.0 * np.pi)
                return -torch.logsumexp(log_probs, dim=-1).mean()

            logits = self.categorical_heads[target_index](context)
            return F.cross_entropy(logits, target.long())

        def sample_feature(self, data, target_index, rng):
            context = self.forward(data, target_index)
            spec = self.column_specs[target_index]
            if spec.is_numeric:
                params = self.numeric_heads[target_index](context)
                logits, means, log_scales = torch.chunk(params, 3, dim=-1)
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                means = means.detach().cpu().numpy()
                scales = np.exp(np.clip(log_scales.detach().cpu().numpy(), -7.0, 5.0))
                sampled = np.empty(len(data), dtype=float)
                for row_index in range(len(data)):
                    component = rng.choice(mixture_components, p=probs[row_index])
                    sampled[row_index] = rng.normal(means[row_index, component], scales[row_index, component])
                return sampled

            logits = self.categorical_heads[target_index](context)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            return np.asarray([rng.choice(len(spec.categories), p=row_probs) for row_probs in probs], dtype=float)

    class _TorchGibbsEstimator:
        def __init__(self):
            self.model = None
            self.device = None

        def fit(self, train_data, column_specs, train_kwargs, sample_seed):
            self.device = torch.device(train_kwargs["device"])
            self.model = _MaskedTransformer(column_specs).to(self.device)
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=float(train_kwargs["lr"]),
                weight_decay=float(train_kwargs["weight_decay"]),
            )
            batch_size = max(1, min(int(train_kwargs["batch_size"]), len(train_data)))
            epochs = int(train_kwargs["epochs"])
            grad_clip = float(train_kwargs["grad_clip"])
            tensor_data = torch.tensor(train_data, dtype=torch.float32, device=self.device)
            rng = np.random.RandomState(sample_seed)
            self.model.train()
            for _ in range(epochs):
                order = rng.permutation(len(train_data))
                for start in range(0, len(order), batch_size):
                    batch_indices = order[start : start + batch_size]
                    target_index = int(rng.randint(len(column_specs)))
                    batch = tensor_data[batch_indices]
                    optimizer.zero_grad()
                    loss = self.model.loss(batch, target_index)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    optimizer.step()
            return self

        def sample(self, n_samples, train_data, column_specs, sample_kwargs, sample_seed):
            if self.model is None:
                raise RuntimeError("Estimator must be fitted before sampling")

            gibbs_rounds = int(sample_kwargs["gibbs_rounds"])
            batch_size = max(1, int(sample_kwargs["batch_size"]))
            rng = np.random.RandomState(sample_seed)
            start_indices = rng.choice(len(train_data), size=n_samples, replace=True)
            samples = np.asarray(train_data[start_indices], dtype=float).copy()
            self.model.eval()
            with torch.no_grad():
                for _ in range(gibbs_rounds):
                    for target_index in rng.permutation(len(column_specs)):
                        for start in range(0, n_samples, batch_size):
                            stop = min(start + batch_size, n_samples)
                            batch = torch.tensor(samples[start:stop], dtype=torch.float32, device=self.device)
                            samples[start:stop, target_index] = self.model.sample_feature(batch, target_index, rng)
            return samples

    return _TorchGibbsEstimator()
