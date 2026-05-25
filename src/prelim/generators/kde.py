import warnings
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import NearestNeighbors
from statsmodels.nonparametric.bandwidths import bw_silverman, bw_scott
from .base import BaseGenerator

# to choose bandwidth via CV, see, for instance, 
# https://scikit-learn.org/stable/auto_examples/neighbors/plot_digits_kde_sampling.html#sphx-glr-auto-examples-neighbors-plot-digits-kde-sampling-py


class Gen_kdebw(BaseGenerator):
    # Global multivariate KDE with a single scalar bandwidth derived from
    # Silverman or Scott per-dimension rules and then averaged.

    def __init__(self, method='silverman', seed=2020):
        super().__init__("kdebw", seed=seed)
        if method == 'silverman':
            self.bw_method_ = bw_silverman
        elif method == 'scott':
            self.bw_method_ = bw_scott
        else:
            raise ValueError("The method must be either scott or silverman")
        self.model_ = None

    def fit(self, X, y=None, metamodel=None):
        bw = self.bw_method_(X)
        if bw.max()/bw.min() > 10:
            warnings.warn("Bandwidths for different dimensions differ by more than order of magnitude. "
                          "Consider using z-score scaling")
        bw = bw.mean()
        self.model_ = KernelDensity(bandwidth=bw).fit(X)
        return self

    def sample(self, n_samples=1):
        return self.model_.sample(n_samples, random_state=self.rng_)


class Gen_classkde(BaseGenerator):
    # Class-conditional version of Gen_kdebw: fit one global multivariate KDE
    # per label. Small classes shrink their scalar bandwidth toward the global
    # bandwidth using alpha = n_cls / (n_cls + c).

    def __init__(self, method='silverman', balanced=False, c=20, seed=2020):
        super().__init__("class_kde", seed=seed)
        if method == 'silverman':
            self.bw_method_ = bw_silverman
        elif method == 'scott':
            self.bw_method_ = bw_scott
        else:
            raise ValueError("The method must be either scott or silverman")
        self.balanced_ = balanced
        self.c_ = c
        self.classes_ = None
        self.priors_ = None
        self.global_bandwidth_ = None
        self.bandwidths_ = None
        self.models_ = None

    def fit(self, X, y=None, metamodel=None):
        del metamodel
        if y is None:
            raise ValueError("Gen_classkde.fit requires y")

        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_, counts = np.unique(y, return_counts=True)
        self.priors_ = counts / counts.sum()
        bw_global = self.bw_method_(X)
        if bw_global.max()/bw_global.min() > 10:
            warnings.warn("Bandwidths for different dimensions differ by more than order of magnitude. "
                          "Consider using z-score scaling")
        self.global_bandwidth_ = bw_global.mean()
        self.bandwidths_ = {}
        self.models_ = {}

        for cls, count in zip(self.classes_, counts):
            Xcls = X[y == cls]
            bw = self.bw_method_(Xcls)
            if bw.max()/bw.min() > 10:
                warnings.warn("Bandwidths for different dimensions differ by more than order of magnitude. "
                              "Consider using z-score scaling")
            alpha = count / (count + self.c_)
            bandwidth = alpha * bw.mean() + (1 - alpha) * self.global_bandwidth_
            self.bandwidths_[cls] = bandwidth
            self.models_[cls] = KernelDensity(bandwidth=bandwidth).fit(Xcls)
        return self

    def _class_counts(self, n_samples):
        if self.balanced_:
            base = n_samples // len(self.classes_)
            counts = np.full(len(self.classes_), base, dtype=int)
            remainder = n_samples - counts.sum()
            if remainder:
                counts[:remainder] += 1
            return counts

        return self.rng_.multinomial(n_samples, self.priors_)

    def sample(self, n_samples=1):
        if self.classes_ is None:
            raise RuntimeError("Gen_classkde.sample called before fit")

        rows = []
        for cls, count in zip(self.classes_, self._class_counts(n_samples)):
            if count == 0:
                continue
            rows.append(self.models_[cls].sample(count, random_state=self.rng_))

        X = np.vstack(rows)
        xdim = X.shape[0]
        return X[self.rng_.choice(np.arange(xdim), size=xdim, replace=False), :].copy()


class Gen_kdebwhl(BaseGenerator):
    # Same KDE fit as Gen_kdebw, but reject sampled rows that leave the
    # observed feature-wise min/max range ("hard limits").

    def __init__(self, method = 'silverman', seed=2020):
        super().__init__("kdebwhl", seed=seed)
        if method == 'silverman':
            self.bw_method_ = bw_silverman
        elif method == 'scott':
            self.bw_method_ = bw_scott
        else:
            raise ValueError("The method must be either scott or silverman")
        self.model_ = None
        self.limits_ = None

    def fit(self, X, y=None, metamodel=None):
        bw = self.bw_method_(X)
        if bw.max()/bw.min() > 10:
            warnings.warn("Bandwidths for different dimensions differ by more than order of magnitude. "
                          "Consider using z-score scaling")
        bw = bw.mean()
        self.model_ = KernelDensity(bandwidth=bw).fit(X)
        self.limits_ = (X.min(axis=0), X.max(axis=0))
        return self

    def sample(self, n_samples = 1):
        return self._generate_w_hard_limits(n_samples)
    
    def _generate_w_hard_limits(self, n_samples):
        sample = self._cleaned_sample(n_samples)
        mult = int(min(20, n_samples/max(sample.shape[0], 10) + 1))
        while sample.shape[0] < n_samples:
            additional = self._cleaned_sample(n_samples * mult)
            sample = np.append(sample, additional, axis = 0)
            if (sample.shape[0]/n_samples < 0.01):
                raise RuntimeError("< 1 % of generated points are within the limits; please make sure you scaled the data")
        return sample[:n_samples]

    def _cleaned_sample(self, n_samples):
        new_samples = self.model_.sample(n_samples, random_state=self.rng_)
        new_samples = new_samples[((new_samples <= self.limits_[1]) & (new_samples >= self.limits_[0])).all(axis = 1)]
        return new_samples


class Gen_kdeb(BaseGenerator):
    # Neighbourhood bootstrap baseline: pick observed rows and perturb them by
    # a random direction, with radius controlled by a k-NN distance heuristic.
    def __init__(self, knn=10, seed=2020):
        super().__init__("kdeb", seed=seed)
        self.knn_ = knn
        self.X_ = None
        self.dist_ = None

    def fit(self, X, y=None, metamodel=None):
        self.X_ = X.copy()
        if self.knn_ == 0:
            self.dist_ = 1
        elif self.knn_ >= X.shape[0]:
            raise RuntimeError(
                "The dataset is too small or the knn value is too large. "
                "Number of data points must be greater than k."
            )
        else:
            self.dist_ = np.mean(NearestNeighbors(n_neighbors=self.knn_).fit(X).kneighbors()[0][:, self.knn_ - 1])
        return self

    def sample(self, n_samples):
        # http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
        d = self.X_.shape[1]
        u = self.rng_.normal(0, 1, (n_samples, d + 2))
        den = np.sum(u**2, axis=1) ** 0.5
        u = u / den[:, None]

        base_rows = self.rng_.choice(self.X_.shape[0], n_samples)
        return self.X_[base_rows, :] + u[:, 0:d]


class Gen_kdebwm(BaseGenerator):
    # Marginal KDE baseline: fit one independent 1D KDE per feature using
    # per-dimension bandwidths, so cross-feature dependence is ignored.

    def __init__(self, method='silverman', seed=2020):
        super().__init__("kdebwm", seed=seed)
        if method == 'silverman':
            self.bw_method_ = bw_silverman
        elif method == 'scott':
            self.bw_method_ = bw_scott
        else:
            raise ValueError("The method must be either scott or silverman")
        self.model_ = None

    def fit(self, X, y=None, metamodel=None):
        bw = self.bw_method_(X)
        self.model_ = []
        for i in range(X.shape[1]):
            self.model_.append(KernelDensity(bandwidth=bw[i]).fit(X[:, i].reshape(-1, 1)))
        return self

    def sample(self, n_samples=1):
        newdata = []
        for i in range(len(self.model_)):
            newdata.append(self.model_[i].sample(n_samples, random_state=self.rng_))
        return np.hstack(newdata)
