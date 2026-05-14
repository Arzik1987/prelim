from math import ceil

import numpy as np
import pandas as pd
from tabgan.sampler import GANGenerator

from .base import BaseGenerator


class Gen_tabgan(BaseGenerator):
    def __init__(self, generator_kwargs: dict | None = None, seed=2020):
        super().__init__("tabgan", seed=seed)
        self.generator_kwargs_ = {} if generator_kwargs is None else dict(generator_kwargs)
        self.X_ = None

    def fit(self, X, y=None, metamodel=None):
        self.X_ = np.asarray(X).copy()
        return self

    def sample(self, n_samples=1):
        train_df = pd.DataFrame(self.X_)
        target = pd.DataFrame({"target": np.zeros(len(train_df), dtype=int)})
        test_df = train_df.copy()

        generator_kwargs = dict(self.generator_kwargs_)
        requested_multiplier = ceil((n_samples * 1.1) / len(train_df))
        generator_kwargs["gen_x_times"] = max(generator_kwargs.get("gen_x_times", 0), requested_multiplier)

        generator = GANGenerator(**generator_kwargs)
        sampled_df, _ = generator.generate_data_pipe(
            train_df=train_df,
            target=target,
            test_df=test_df,
            only_generated_data=True,
            use_adversarial=False,
        )
        sampled = sampled_df.to_numpy()
        if sampled.shape[0] < n_samples:
            if sampled.shape[0] == 0:
                raise RuntimeError("TabGAN returned no generated rows")
            missing = n_samples - sampled.shape[0]
            pad_indices = self.rng_.choice(sampled.shape[0], size=missing, replace=True)
            sampled = np.concatenate([sampled, sampled[pad_indices]], axis=0)

        return sampled[:n_samples, :]
