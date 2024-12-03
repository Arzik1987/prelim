import numpy as np
import scipy.stats as stats
from prelim.generators.munge import Gen_munge
from prelim.generators.adasyn import GenAdasyn
from prelim.generators.dummy import Gen_dummy
from prelim.generators.kde import Gen_kdebw, Gen_kdebwhl
from prelim.generators.kdeb import Gen_kdeb
from prelim.generators.kdem import Gen_kdebwm

from test.dummy_metamodel import DummyMeta

import pytest

from prelim.generators.noise import Gen_noise
from prelim.generators.perfect import Gen_perfect
from prelim.generators.rerx import Gen_rerx
from prelim.generators.rfdens import Gen_rfdens
from prelim.generators.smote import Gen_smote


def test_adasyn():
    generator = GenAdasyn()
    sample_size = 100
    alpha = 0.001
    assert _valid_2d_dist(generator, sample_size, alpha), "Valid 2d distribution"
    assert not _invalid_2d_dist(generator, sample_size, alpha), "Invalid 2d distribution"


def test_dummy():
    generator = Gen_dummy()
    sample_size = 100
    alpha = 0.001
    assert _valid_2d_dist(generator, sample_size, alpha), "Valid 2d distribution"
    assert not _invalid_2d_dist(generator, sample_size, alpha), "Invalid 2d distribution"


def test_kde():
    generator1 = Gen_kdebw()
    generator2 = Gen_kdebwhl()
    sample_size = 100
    alpha = 0.001
    assert _valid_2d_dist(generator1, sample_size, alpha), "Valid 2d distribution"
    assert not _invalid_2d_dist(generator1, sample_size, alpha), "Valid 2d distribution"

    assert _valid_2d_dist(generator2, sample_size, alpha), "Valid 2d distribution"
    assert not _invalid_2d_dist(generator2, sample_size, alpha), "Invalid 2d distribution"


def test_kdeb():
    generator = Gen_kdeb()
    sample_size = 100
    alpha = 0.001
    assert _valid_2d_dist(generator, sample_size, alpha), "Valid 2d distribution"
    assert not _invalid_2d_dist(generator, sample_size, alpha), "Invalid 2d distribution"


def test_kdem():
    generator = Gen_kdebwm()
    sample_size = 100
    alpha = 0.001
    assert _valid_2d_dist(generator, sample_size, alpha), "Valid 2d distribution"
    assert not _invalid_2d_dist(generator, sample_size, alpha), "Invalid 2d distribution"


def test_munge_():
    generator = Gen_munge()
    sample_size = 100
    alpha = 0.001
    assert _valid_2d_dist(generator, sample_size, alpha), "Valid 2d normal"
    assert not _invalid_2d_dist(generator, sample_size, alpha), "Invalid 2d normal"


def test_rerx():
    generator = Gen_rerx()
    metamodel = DummyMeta()
    sample_size = 100
    y = np.repeat(1, sample_size)
    alpha = 0.001
    assert _valid_2d_dist(generator, sample_size, alpha, metamodel=metamodel, y=y), "Valid 2d normal"
    assert not _invalid_2d_dist(generator, sample_size, alpha, metamodel=metamodel, y=y), "Invalid 2d normal"


def test_rfdens():
    generator = Gen_rfdens()
    sample_size = 100
    y = np.repeat(1, sample_size)
    alpha = 0.0001
    assert _valid_2d_dist(generator, sample_size, alpha, y=y), "Valid 2d normal"
    assert not _invalid_2d_dist(generator, sample_size, alpha, y=y), "Invalid 2d normal"


def test_smote():
    generator = Gen_smote()
    sample_size = 100
    alpha = 0.001
    assert _valid_2d_dist(generator, sample_size, alpha), "Valid 2d normal"
    assert not _invalid_2d_dist(generator, sample_size, alpha), "Invalid 2d normal"


def test_noise_():
    generator = Gen_noise()
    sample_size = 500
    alpha = 0.5
    assert not _valid_2d_dist(generator, sample_size, alpha), "Valid 2d normal"
    assert not _invalid_2d_dist(generator, sample_size, alpha), "Invalid 2d normal"


def test_rand_():
    generator = Gen_munge()
    sample_size = 500
    alpha = 0.5
    assert not _valid_2d_dist(generator, sample_size, alpha), "Valid 2d normal"
    assert not _invalid_2d_dist(generator, sample_size, alpha), "Invalid 2d normal"


def test_perfect_valid():

    np.random.seed(2020)

    generator = Gen_perfect()
    sample_size = 100
    alpha = 0.001

    # distribution
    means = [0, 0]
    vars_ = [0.5, 0.5]
    covs = np.diag(vars_)
    sample = np.random.multivariate_normal(means, covs, sample_size)

    generator.fit(X=sample)
    est_sample = generator.sample(sample_size - 1)
    est_means = np.mean(est_sample, axis=0)

    zs = _gauss_stat(means, vars_, est_means, sample_size)  # std normal gauß = est_means

    std_norm = lambda v: (stats.norm.pdf(v[0], loc=0, scale=1), stats.norm.pdf(v[1], loc=0, scale=1))
    alphas = np.asarray(2 * [alpha])
    assert np.all(np.abs(std_norm(zs)) > alphas), "Valid 2d dist for Perfect Generator"


def test_perfect_invalid():

    np.random.seed(2020)

    alpha = 0.5
    sample_size = 100
    generator= Gen_perfect()

    # test distribution
    means_1 = [0, 0]
    vars_1 = [0.005, 0.005]

    # generator distribution
    means_2 = [100, 100]
    vars_2 = [1, 1]
    covs_2 = np.diag(vars_2)
    sample_2 = np.random.multivariate_normal(means_2, covs_2, sample_size)

    generator.fit(X=sample_2)
    est_sample = generator.sample(sample_size - 1)
    est_means = np.mean(est_sample, axis=0)

    zs = _gauss_stat(means_1, vars_1, est_means, sample_size)  # std normal gauß = est_means

    std_norm = lambda v: (stats.norm.pdf(v[0], loc=0, scale=1), stats.norm.pdf(v[1], loc=0, scale=1))
    alphas = np.asarray(2 * [alpha])
    assert not np.all(np.abs(std_norm(zs)) > alphas), "Perfect Generator Invalid dist"


def _valid_2d_dist(generator, sample_size, alpha, y=None, metamodel=None):

    np.random.seed(2020)

    # distribution
    means = [0, 0]
    vars_ = [0.5, 0.5]
    covs = np.diag(vars_)
    sample = np.random.multivariate_normal(means, covs, sample_size)

    generator.fit(X=sample, y=y, metamodel=metamodel)
    est_sample = generator.sample(sample_size)
    est_means = np.mean(est_sample, axis=0)

    zs = _gauss_stat(means, vars_, est_means, sample_size)  # std normal gauß = est_means

    std_norm = lambda v: (stats.norm.pdf(v[0], loc=0, scale=1), stats.norm.pdf(v[1], loc=0, scale=1))
    result = False

    alphas = np.asarray(2 * [alpha])   # two-sided
    if np.all(np.abs(std_norm(zs)) > alphas):
        result = True
    return result


def _invalid_2d_dist(generator, sample_size, alpha, y=None, metamodel=None):

    np.random.seed(2020)

    # test distribution
    means_1 = [0, 0]
    vars_1 = [0.005, 0.005]

    # generator distribution
    means_2 = [100, 100]
    vars_2 = [1, 1]
    covs_2 = np.diag(vars_2)
    sample_2 = np.random.multivariate_normal(means_2, covs_2, sample_size)

    generator.fit(X=sample_2, y=y, metamodel=metamodel)
    est_sample = generator.sample(sample_size)
    est_means = np.mean(est_sample, axis=0)

    zs = _gauss_stat(means_1, vars_1, est_means, sample_size)  # std normal gauß = est_means

    std_norm = lambda v: (stats.norm.pdf(v[0], loc=0, scale=1), stats.norm.pdf(v[1], loc=0, scale=1))
    result = False

    alphas = np.asarray(2 * [1/2 * alpha])  # two-sided
    if np.all(np.abs(std_norm(zs)) > alphas):
        result = True
    return result


def _gauss_stat(mean, var, est_mean, n):
    est_mean = np.asarray(est_mean)
    var = np.asarray(var)
    mean = np.asarray(mean)
    return np.sqrt(n) * (est_mean - mean) / np.sqrt(var)
