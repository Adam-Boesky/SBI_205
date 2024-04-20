"""My custom distributions."""
import numpy as np
import torch
import torch.distributions as dist

from scipy import stats
from torch.distributions import constraints, Distribution, Normal


class TenExpTransform(dist.transforms.Transform):
    """
    A transformation with the forward operation applying 10^x.
    """
    domain = dist.constraints.real
    codomain = dist.constraints.positive
    bijective = True
    sign = 1
    
    def __eq__(self, other):
        return type(self) == type(other)

    def _call(self, x):
        return 10**x

    def _inverse(self, y):
        return torch.log10(y)

    def log_abs_det_jacobian(self, x, y):
        """
        Return the log determinant of the absolute value of the Jacobian matrix.
        """
        return torch.log(torch.tensor(10.0)) * y


class LogUniform(dist.TransformedDistribution):
    def __init__(self, low: torch.Tensor, high: torch.Tensor):
        self.low = low.log10()
        self.high = high.log10()
        super().__init__(dist.Uniform(low.log10(), high.log10()), TenExpTransform())

    @property
    def support(self):
        return constraints.interval(self.low, self.high)


class TruncatedNormal(Distribution):
    def __init__(self, lower, upper, mu, sigma, validate_args=None):
        self.lower = lower
        self.upper = upper
        self.mu = mu
        self.sigma = sigma
        self.normal = Normal(mu, sigma)

        # Pre-compute the a and b parameters for scipy's truncnorm
        self.a = (self.lower - self.mu) / self.sigma
        self.b = (self.upper - self.mu) / self.sigma
        
        # Calculate normalization constant in log space for numerical stability
        self.Z = (self.normal.cdf(self.upper) - self.normal.cdf(self.lower)).log()
        
        super(TruncatedNormal, self).__init__(batch_shape=self.mu.size(), event_shape=torch.Size(), validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        if not isinstance(sample_shape, tuple):
            sample_shape = (sample_shape,)

        # Generate samples using scipy's truncnorm
        total_shape = sample_shape + self.batch_shape
        total_num_samples = np.prod(total_shape).item()
        samples_np = stats.truncnorm.rvs(self.a, self.b, loc=self.mu.item(), scale=self.sigma.item(), size=total_num_samples)
        samples = torch.from_numpy(samples_np).reshape(total_shape).to(dtype=self.mu.dtype)

        return samples

    def log_prob(self, value):
        log_prob = self.normal.log_prob(value) - self.Z
        mask = (value < self.lower) | (value > self.upper)
        log_prob[mask] = float('-inf')
        return log_prob

    @property
    def support(self):
        return constraints.interval(self.lower, self.upper)

    @property
    def mean(self):
        # Placeholder for actual mean calculation of truncated normal
        return self.mu

    @property
    def variance(self):
        # Placeholder for actual variance calculation of truncated normal
        return self.sigma**2
