import numpy as np
import pytest

from prelim.generators.adasyn import Gen_adasyn
from prelim.generators.dummy import Gen_dummy
from prelim.generators.gmm import Gen_gmm, Gen_gmmbic, Gen_gmmbical
from prelim.generators.kde import Gen_kdebw, Gen_kdebwhl
from prelim.generators.kdeb import Gen_kdeb
from prelim.generators.kdem import Gen_kdebwm
from prelim.generators.munge import Gen_munge
from prelim.generators.noise import Gen_noise
from prelim.generators.perfect import Gen_perfect
from prelim.generators.rand import Gen_randn, Gen_randu
from prelim.generators.rerx import Gen_rerx
from prelim.generators.rfdens import Gen_rfdens
from prelim.generators.smote import Gen_smote
from prelim.generators.vva import Gen_vva as Gen_vva_legacy
from prelim.generators.vva_p import Gen_vva as Gen_vva_proba


def _clustered_sample():
    rng = np.random.RandomState(2020)
    x1 = rng.multivariate_normal([0.0, 0.0], np.eye(2) * 0.2, 40)
    x2 = rng.multivariate_normal([3.0, 3.0], np.eye(2) * 0.2, 40)
    return np.vstack((x1, x2))


def _grid_sample():
    return np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [2.0, 2.0],
        ]
    )


def _labeled_clustered_sample():
    x = _clustered_sample()
    y = np.concatenate((np.zeros(40, dtype=int), np.ones(40, dtype=int)))
    return x, y


class _AllOnesMeta:
    def predict(self, x):
        return np.ones(len(x), dtype=int)


class _LinearProbabilityMeta:
    classes_ = np.array([0, 1])

    def predict_proba(self, x):
        p1 = np.clip(0.5 + x[:, 0] / 6.0, 0.01, 0.99)
        return np.column_stack((1.0 - p1, p1))


class _LinearScoreMeta:
    def predict_proba(self, x):
        return np.clip(0.5 + x[:, 0] / 6.0, 0.01, 0.99)


class _SingleSideProbabilityMeta:
    classes_ = np.array([0, 1])

    def predict_proba(self, x):
        p1 = np.full(len(x), 0.9)
        return np.column_stack((1.0 - p1, p1))


class _SingleSideScoreMeta:
    def predict_proba(self, x):
        return np.full(len(x), 0.9)


def test_dummy_returns_full_copy_of_fitted_data():
    x = _clustered_sample()
    generator = Gen_dummy().fit(x)

    sample = generator.sample(n_samples=5)

    assert np.array_equal(sample, x)
    assert sample is not x
    sample[0, 0] = -999.0
    assert generator.X_[0, 0] != -999.0
    assert generator.my_name() == "dummy"


def test_perfect_returns_subset_without_replacement_when_possible():
    x = _clustered_sample()
    generator = Gen_perfect().fit(x)

    sample = generator.sample(n_samples=10)

    assert sample.shape == (10, x.shape[1])
    assert len(np.unique(sample, axis=0)) == 10
    assert set(map(tuple, sample)).issubset(set(map(tuple, x)))
    assert generator.my_name() == "perfect"


def test_perfect_warns_and_returns_complete_set_when_too_many_points_requested():
    x = _clustered_sample()
    generator = Gen_perfect().fit(x)

    with pytest.warns(UserWarning, match="Too many points are requested"):
        sample = generator.sample(n_samples=len(x) + 1)

    assert np.array_equal(sample, x)


def test_noise_perturbs_points_within_expected_per_feature_range():
    x = _grid_sample()
    generator = Gen_noise(scale=0.3).fit(x)

    sample = generator.sample(n_samples=1000)

    assert sample.shape == x.shape
    assert generator.my_name() == "noise"
    max_delta = 0.3
    assert np.all(np.abs(sample - x) <= max_delta + 1e-12)


def test_randu_samples_stay_within_feature_bounds():
    x = _clustered_sample()
    generator = Gen_randu().fit(x)

    sample = generator.sample(n_samples=25)

    assert sample.shape == (25, x.shape[1])
    assert np.all(sample >= x.min(axis=0))
    assert np.all(sample <= x.max(axis=0))
    assert generator.my_name() == "randu"


def test_randn_learns_location_and_covariance_and_samples_requested_shape():
    x = _clustered_sample()
    generator = Gen_randn(seed=2020).fit(x)

    sample = generator.sample(n_samples=25)

    assert sample.shape == (25, x.shape[1])
    assert generator.location_.shape == (x.shape[1],)
    assert generator.covariance_.shape == (x.shape[1], x.shape[1])
    assert generator.my_name() == "randn"


@pytest.mark.parametrize(
    ("generator_a", "generator_b", "sample_kwargs"),
    [
        (Gen_randn(seed=2020), Gen_randn(seed=2020), {"n_samples": 25}),
        (Gen_randu(seed=2020), Gen_randu(seed=2020), {"n_samples": 25}),
        (Gen_noise(scale=0.3, seed=2020), Gen_noise(scale=0.3, seed=2020), {"n_samples": 25}),
        (Gen_perfect(seed=2020), Gen_perfect(seed=2020), {"n_samples": 10}),
        (Gen_kdeb(knn=5, seed=2020), Gen_kdeb(knn=5, seed=2020), {"n_samples": 20}),
        (Gen_kdebw(seed=2020), Gen_kdebw(seed=2020), {"n_samples": 20}),
        (Gen_kdebwhl(seed=2020), Gen_kdebwhl(seed=2020), {"n_samples": 20}),
        (Gen_kdebwm(seed=2020), Gen_kdebwm(seed=2020), {"n_samples": 20}),
        (Gen_munge(local_var=1, p_swap=0.5, seed=2020), Gen_munge(local_var=1, p_swap=0.5, seed=2020), {"n_samples": 20}),
        (Gen_rfdens(seed=2020), Gen_rfdens(seed=2020), {"n_samples": 20}),
        (Gen_vva_proba(seed=2020), Gen_vva_proba(seed=2020), {"r": 1.0}),
    ],
)
def test_seeded_generators_are_reproducible(generator_a, generator_b, sample_kwargs):
    x = _clustered_sample()
    y = np.concatenate((np.zeros(40, dtype=int), np.ones(40, dtype=int)))

    if isinstance(generator_a, Gen_rfdens):
        generator_a.fit(x, y)
        generator_b.fit(x, y)
    elif isinstance(generator_a, Gen_vva_proba):
        generator_a.fit(x, _LinearProbabilityMeta())
        generator_b.fit(x, _LinearProbabilityMeta())
    else:
        generator_a.fit(x)
        generator_b.fit(x)

    sample_a = generator_a.sample(**sample_kwargs)
    sample_b = generator_b.sample(**sample_kwargs)

    assert np.allclose(sample_a, sample_b)


def test_kde_hard_limits_samples_within_observed_min_max():
    x = _clustered_sample()
    generator = Gen_kdebwhl().fit(x)

    sample = generator.sample(n_samples=50)

    assert sample.shape == (50, x.shape[1])
    assert np.all(sample >= x.min(axis=0))
    assert np.all(sample <= x.max(axis=0))
    assert generator.my_name() == "kdebwhl"


def test_kde_bandwidth_generator_returns_requested_shape():
    x = _clustered_sample()
    generator = Gen_kdebw().fit(x)

    sample = generator.sample(n_samples=50)

    assert sample.shape == (50, x.shape[1])
    assert generator.my_name() == "kdebw"


def test_kdeb_rejects_knn_greater_than_or_equal_to_dataset_size():
    x = _clustered_sample()

    with pytest.raises(RuntimeError, match="dataset is too small"):
        Gen_kdeb(knn=len(x)).fit(x)


def test_kdeb_samples_requested_shape_for_knn_zero_example():
    x = np.array([[0.0, 0.0]])
    generator = Gen_kdeb(knn=0).fit(x)

    sample = generator.sample(n_samples=20)

    assert sample.shape == (20, x.shape[1])
    assert generator.dist_ == 1
    assert generator.my_name() == "kdeb"


def test_kdebwm_invalid_method_is_rejected():
    with pytest.raises(ValueError, match="either scott or silverman"):
        Gen_kdebwm(method="invalid")


def test_kdebwm_samples_requested_shape():
    x = _clustered_sample()
    generator = Gen_kdebwm().fit(x)

    sample = generator.sample(n_samples=30)

    assert sample.shape == (30, x.shape[1])
    assert len(generator.model_) == x.shape[1]
    assert generator.my_name() == "kdebwm"


def test_munge_rejects_too_small_p_swap():
    with pytest.raises(SystemExit, match="p_swap parameter is too small"):
        Gen_munge(p_swap=0.001)


def test_munge_generates_unique_requested_number_of_rows():
    x = _clustered_sample()
    generator = Gen_munge(local_var=1, p_swap=0.5, seed=2020).fit(x)

    sample = generator.sample(n_samples=60)

    assert sample.shape == (60, x.shape[1])
    assert len(np.unique(sample, axis=0)) == 60
    assert generator.my_name() == "munge"


@pytest.mark.parametrize(
    ("generator_cls", "expected_name"),
    [
        (Gen_smote, "smote"),
        (Gen_adasyn, "adasyn"),
    ],
)
def test_smote_like_generators_warn_when_requested_size_is_smaller_than_train_set(generator_cls, expected_name):
    x = _clustered_sample()
    generator = generator_cls().fit(x)

    with pytest.warns(UserWarning):
        sample = generator.sample(n_samples=20)

    assert sample.shape == (20, x.shape[1])
    assert generator.my_name() in {expected_name, "adasyns"}


@pytest.mark.parametrize(
    ("generator_cls", "expected_name"),
    [
        (Gen_smote, "smote"),
        (Gen_adasyn, "adasyn"),
    ],
)
def test_smote_like_generators_return_requested_shape_on_example_style_input(generator_cls, expected_name):
    x = _clustered_sample()
    generator = generator_cls().fit(x)

    sample = generator.sample(n_samples=120)

    assert sample.shape == (120, x.shape[1])
    assert generator.my_name() in {expected_name, "adasyns"}


def test_rerx_returns_only_correctly_predicted_rows():
    x, y = _labeled_clustered_sample()
    generator = Gen_rerx().fit(x, y, _AllOnesMeta())

    sample = generator.sample(n_samples=5)

    assert np.array_equal(sample, x[y == 1])
    assert generator.my_name() == "rerx"


def test_rfdens_fit_populates_boxes_and_sample_stays_within_global_bounds():
    x, y = _labeled_clustered_sample()
    generator = Gen_rfdens()

    generator.fit(x, y)
    sample = generator.sample(n_samples=25)

    assert len(generator.boxes_) > 0
    assert len(generator.boxes_) == len(generator.nsamples_)
    assert sample.shape == (25, x.shape[1])
    assert np.all(sample >= x.min(axis=0))
    assert np.all(sample <= x.max(axis=0))
    assert generator.my_name() == "cmmrf"


@pytest.mark.parametrize(
    ("generator_cls", "expected_name"),
    [
        (Gen_gmm, "gmmcv"),
        (Gen_gmmbic, "gmm"),
        (Gen_gmmbical, "gmmal"),
    ],
)
def test_gmm_family_generates_requested_shape(generator_cls, expected_name):
    x = _clustered_sample()
    params = {"covariance_type": ["diag"], "n_components": [1, 2]} if generator_cls is not Gen_gmmbical else {"n_components": [1, 2]}
    generator = generator_cls(params=params).fit(x)

    sample = generator.sample(n_samples=20)

    assert sample.shape == (20, x.shape[1])
    assert generator.my_name() == expected_name


def test_vva_proba_returns_empty_sample_for_r_zero_and_out_of_range_r_fails():
    x = _clustered_sample()
    generator = Gen_vva_proba().fit(x, _LinearProbabilityMeta())

    empty = generator.sample(r=0)

    assert empty.shape == (0, x.shape[1])
    with pytest.raises(ValueError, match="from 0 to 2.5"):
        generator.sample(r=3.0)
    assert generator.my_name() == "vva"


def test_vva_proba_disables_generation_when_all_predictions_are_on_one_side():
    x = _clustered_sample()
    generator = Gen_vva_proba().fit(x, _SingleSideProbabilityMeta())

    assert generator.will_generate() is False
    assert generator.sample(r=1.0).shape == (0, x.shape[1])


def test_vva_proba_generates_requested_number_of_boundary_points():
    x = _clustered_sample()
    generator = Gen_vva_proba(rho=0.2).fit(x, _LinearProbabilityMeta())

    sample = generator.sample(r=1.0)

    assert generator.will_generate() is True
    assert sample.shape == (len(x), x.shape[1])


def test_vva_legacy_disables_generation_when_all_predictions_are_on_one_side():
    x = _clustered_sample()
    generator = Gen_vva_legacy().fit(x, _SingleSideScoreMeta())

    assert generator.will_generate() is False
    assert generator.sample(r=1.0).shape == (0, x.shape[1])


def test_vva_legacy_generates_requested_number_of_boundary_points():
    x = _clustered_sample()
    generator = Gen_vva_legacy(rho=0.2).fit(x, _LinearScoreMeta())

    sample = generator.sample(r=1.0)

    assert generator.will_generate() is True
    assert sample.shape == (len(x), x.shape[1])
