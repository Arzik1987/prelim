import numpy as np
import pandas as pd

from .base import BaseGenerator

try:
    from pgmpy.estimators import BIC, BayesianEstimator, HillClimbSearch
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.sampling import BayesianModelSampling
except Exception:
    BIC = None
    BayesianEstimator = None
    HillClimbSearch = None
    DiscreteBayesianNetwork = None
    BayesianModelSampling = None


def _fit_network(model, data, estimator_class):
    try:
        from pgmpy.parameter_estimator import DiscreteMLE
    except Exception:
        return model.fit(data, estimator=estimator_class)
    return model.fit(data, estimator=DiscreteMLE())


class Gen_bayesnet(BaseGenerator):
    def __init__(self, max_bins=8, search_kwargs: dict | None = None, seed=2020):
        super().__init__("bayesnet", seed=seed)
        self.max_bins_ = max_bins
        self.search_kwargs_ = {} if search_kwargs is None else dict(search_kwargs)
        self.X_ = None
        self.column_specs_ = None
        self.columns_ = None
        self.model_ = None
        self.sampler_ = None

    def fit(self, X, y=None, metamodel=None):
        self.X_ = np.asarray(X).copy()
        discrete_df, self.column_specs_ = self._discretize(self.X_)
        self.columns_ = list(discrete_df.columns)

        score_class = BIC
        estimator_class = BayesianEstimator
        search_class = HillClimbSearch
        model_class = DiscreteBayesianNetwork
        sampler_class = BayesianModelSampling
        if None in (score_class, estimator_class, search_class, model_class, sampler_class):
            from pgmpy.estimators import BIC as score_class
            from pgmpy.estimators import BayesianEstimator as estimator_class
            from pgmpy.estimators import HillClimbSearch as search_class
            from pgmpy.models import DiscreteBayesianNetwork as model_class
            from pgmpy.sampling import BayesianModelSampling as sampler_class

        scorer = score_class(discrete_df)
        search = search_class(discrete_df)
        dag = search.estimate(scoring_method=scorer, show_progress=False, **self.search_kwargs_)

        self.model_ = model_class(dag.edges())
        self.model_.add_nodes_from(self.columns_)
        _fit_network(self.model_, discrete_df, estimator_class)
        self.sampler_ = sampler_class(self.model_)
        return self

    def sample(self, n_samples=1):
        sampled = self.sampler_.forward_sample(size=n_samples, seed=self.seed_, show_progress=False)
        rebuilt = [
            self._rebuild_column(sampled[column].to_numpy(), spec)
            for column, spec in zip(self.columns_, self.column_specs_)
        ]
        return np.column_stack(rebuilt)

    def _discretize(self, X):
        columns = {}
        specs = []

        for index in range(X.shape[1]):
            column = X[:, index].astype(float)
            unique_values = np.unique(column)
            column_name = f"x{index}"

            if unique_values.size == 1:
                columns[column_name] = np.zeros(len(column), dtype=int)
                specs.append({"kind": "constant", "value": float(unique_values[0])})
                continue

            if unique_values.size <= self.max_bins_:
                labels = np.searchsorted(unique_values, column)
                columns[column_name] = labels.astype(int)
                specs.append({"kind": "values", "values": unique_values.astype(float)})
                continue

            edges = np.quantile(column, np.linspace(0.0, 1.0, self.max_bins_ + 1))
            edges = np.unique(edges)
            if edges.size <= 2:
                edges = np.linspace(column.min(), column.max(), self.max_bins_ + 1)

            labels = np.digitize(column, edges[1:-1], right=False)
            labels = np.clip(labels, 0, len(edges) - 2)
            columns[column_name] = labels.astype(int)
            specs.append({"kind": "binned", "edges": edges.astype(float)})

        return pd.DataFrame(columns), specs

    def _rebuild_column(self, labels, spec):
        if spec["kind"] == "constant":
            return np.full(len(labels), spec["value"], dtype=float)

        label_array = np.asarray(labels, dtype=int)
        if spec["kind"] == "values":
            values = spec["values"]
            clipped = np.clip(label_array, 0, len(values) - 1)
            return values[clipped]

        edges = spec["edges"]
        clipped = np.clip(label_array, 0, len(edges) - 2)
        low = edges[clipped]
        high = edges[clipped + 1]
        same_bounds = np.isclose(low, high)
        sampled = self.rng_.uniform(low, high)
        sampled[same_bounds] = low[same_bounds]
        return sampled
