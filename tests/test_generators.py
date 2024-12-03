import numpy as np
from prelim.generators.adasyn import GenAdasyn
from prelim.generators.smote import GenSmote
from prelim.generators.dummy import GenDummy
from test_utils import valid_2d_dist, invalid_2d_dist
import pytest


@pytest.mark.parametrize("GeneratorClass,expected_name", [
    (GenAdasyn, "adasyn"),
    (GenSmote, "smote"),
    (GenDummy, "dummy")
])
def test_generator_fit(GeneratorClass, expected_name):
    X = np.random.rand(100, 5)
    generator = GeneratorClass()
    generator.fit(X)
    assert generator.X_ is not None
    assert generator.my_name() == expected_name


@pytest.mark.parametrize("GeneratorClass", [GenAdasyn, GenSmote])
def test_generator_sample(GeneratorClass):
    X = np.random.rand(100, 5)
    generator = GeneratorClass()
    generator.fit(X)
    samples = generator.sample(n_samples=101)
    assert samples.shape == (101, 5)


def test_dummy_sample():
    X = np.random.rand(100, 5)
    generator = GenDummy()
    generator.fit(X)
    samples = generator.sample(n_samples=101)
    assert samples.shape == (100, 5)


@pytest.mark.parametrize("GeneratorClass", [GenAdasyn, GenSmote,GenDummy])
def test_generator_distributions(GeneratorClass):
    generator = GeneratorClass()
    sample_size = 100
    alpha = 0.001

    # Valid 2D distribution test
    assert valid_2d_dist(generator, sample_size, alpha), "Valid 2D distribution test failed."
    
    # Invalid 2D distribution test
    assert not invalid_2d_dist(generator, sample_size, alpha), "Invalid 2D distribution test failed."


