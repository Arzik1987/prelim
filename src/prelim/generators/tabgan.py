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
        target = pd.Series(np.zeros(len(train_df), dtype=int))
        test_df = train_df.copy()

        generated_batches = []
        generated_rows = 0
        while generated_rows < n_samples:
            generator = GANGenerator(**self.generator_kwargs_)
            sampled_df, _ = generator.generate_data_pipe(
                train_df=train_df,
                target=target,
                test_df=test_df,
                only_generated_data=True,
                use_adversarial=False,
            )
            sampled = sampled_df.to_numpy()
            if sampled.shape[0] == 0:
                raise RuntimeError("TabGAN returned no generated rows")
            generated_batches.append(sampled)
            generated_rows += sampled.shape[0]

        return np.concatenate(generated_batches, axis=0)[:n_samples, :]
