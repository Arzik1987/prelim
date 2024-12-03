import numpy as np
from scipy import stats

def valid_2d_dist(generator, sample_size, alpha, y=None, metamodel=None):
    """
    Check if the generator produces a valid 2D distribution.

    Args:
        generator: The generator instance.
        sample_size: Number of samples to generate.
        alpha: Confidence threshold for validation.
        y: Optional labels.
        metamodel: Optional meta-model for fitting.

    Returns:
        bool: True if the generated distribution matches the expected one, False otherwise.
    """
    np.random.seed(2020)

    # Define the expected distribution
    means = [0, 0]
    vars_ = [0.5, 0.5]
    covs = np.diag(vars_)
    sample = np.random.multivariate_normal(means, covs, sample_size)

    # Fit the generator and generate samples
    generator.fit(X=sample, y=y, metamodel=metamodel)
    est_sample = generator.sample(sample_size)
    est_means = np.mean(est_sample, axis=0)

    # Compute Gaussian statistics
    zs = _gauss_stat(means, vars_, est_means, sample_size)

    # Standard normal distribution function
    std_norm = lambda v: (stats.norm.pdf(v[0], loc=0, scale=1), stats.norm.pdf(v[1], loc=1, scale=1))
    alphas = np.asarray(2 * [alpha])  # two-sided alpha
    return np.all(np.abs(std_norm(zs)) > alphas)


def invalid_2d_dist(generator, sample_size, alpha, y=None, metamodel=None):
    """
    Check if the generator fails to match an invalid 2D distribution.

    Args:
        generator: The GenAdasyn generator instance.
        sample_size: Number of samples to generate.
        alpha: Confidence threshold for validation.
        y: Optional labels.
        metamodel: Optional meta-model for fitting.

    Returns:
        bool: True if the generated distribution does not match, False otherwise.
    """
    np.random.seed(2020)

    # Define a mismatched distribution
    means_1 = [0, 0]
    vars_1 = [0.005, 0.005]

    # Define the generator's assumed distribution
    means_2 = [100, 100]
    vars_2 = [1, 1]
    covs_2 = np.diag(vars_2)
    sample_2 = np.random.multivariate_normal(means_2, covs_2, sample_size)

    # Fit the generator and generate samples
    generator.fit(X=sample_2, y=y, metamodel=metamodel)
    est_sample = generator.sample(sample_size)
    est_means = np.mean(est_sample, axis=0)

    # Compute Gaussian statistics
    zs = _gauss_stat(means_1, vars_1, est_means, sample_size)

    # Standard normal distribution function
    std_norm = lambda v: (stats.norm.pdf(v[0], loc=0, scale=1), stats.norm.pdf(v[1], loc=1, scale=1))
    alphas = np.asarray(2 * [1 / 2 * alpha])  # two-sided alpha
    return np.all(np.abs(std_norm(zs)) > alphas)


def _gauss_stat(mean, var, est_mean, n):
    """
    Compute Gaussian test statistic for hypothesis testing.

    Args:
        mean: True means of the distribution.
        var: Variances of the distribution.
        est_mean: Estimated means from the generator.
        n: Number of samples.

    Returns:
        ndarray: Gaussian test statistics.
    """
    est_mean = np.asarray(est_mean)
    var = np.asarray(var)
    mean = np.asarray(mean)
    return np.sqrt(n) * (est_mean - mean) / np.sqrt(var)