"""
基于NeuralForest的改进实现 / Improved Implementation Based on NeuralForest

中文说明：
本代码基于开源库NeuralForest进行开发，在原有框架基础上实现了创新性改进。
主要贡献包括
- 改进了损失函数设计，增加了后验正则化机制
- 优化了混合高斯模型的参数估计方法

致谢原始库的贡献者，本实现在遵循开源协议的前提下进行了扩展和优化。

English Description:
This code is developed based on the open-source NeuralForest library, implementing 
innovative improvements upon the original framework.
Main contributions include: [Please fill in your specific innovations, e.g.:]
- Enhanced loss function design with posterior regularization mechanism
- Optimized parameter estimation methods for Gaussian Mixture Models

We acknowledge the contributors of the original library. This implementation extends 
and optimizes the framework while complying with open-source licenses.


原始库链接 / Original Library:
GitHub: https://github.com/Nixtla/neuralforecast

使用许可 / License:
本代码遵循[具体许可证]许可证 / This code follows [specific license] license
"""
from typing import Optional, Union, Tuple, List
import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, MixtureSameFamily, Distribution

def _divide_no_nan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Auxiliary funtion to handle divide by 0
    """
    div = a / b
    return torch.nan_to_num(div, nan=0.0, posinf=0.0, neginf=0.0)

def _weighted_mean(losses, weights):
    """
    Compute weighted mean of losses per datapoint.
    """
    return _divide_no_nan(torch.sum(losses * weights), torch.sum(weights))

class BasePointLoss(torch.nn.Module):
    """
    Base class for point loss functions.

    **Parameters:**<br>
    `horizon_weight`: Tensor of size h, weight for each timestamp of the forecasting window. <br>
    `outputsize_multiplier`: Multiplier for the output size. <br>
    `output_names`: Names of the outputs. <br>
    """

    def __init__(
        self, horizon_weight=None, outputsize_multiplier=None, output_names=None
    ):
        super(BasePointLoss, self).__init__()
        if horizon_weight is not None:
            horizon_weight = torch.Tensor(horizon_weight.flatten())
        self.horizon_weight = horizon_weight
        self.outputsize_multiplier = outputsize_multiplier
        self.output_names = output_names
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        """
        Input:
        Univariate: [B, H, 1]
        Multivariate: [B, H, N]

        Output: [B, H, N]
        """
        return y_hat

    def _compute_weights(self, y, mask):
        """
        Compute final weights for each datapoint (based on all weights and all masks)
        Set horizon_weight to a ones[H] tensor if not set.
        If set, check that it has the same length as the horizon in x.
        """
        if mask is None:
            mask = torch.ones_like(y)

        if self.horizon_weight is None:
            weights = torch.ones_like(mask)
        else:
            assert mask.shape[1] == len(
                self.horizon_weight
            ), "horizon_weight must have same length as Y"
            weights = self.horizon_weight.clone()
            weights = weights[None, :, None].to(mask.device)
            weights = torch.ones_like(mask, device=mask.device) * weights

        return weights * mask

def level_to_outputs(level):
    qs = sum([[50 - l / 2, 50 + l / 2] for l in level], [])
    output_names = sum([[f"-lo-{l}", f"-hi-{l}"] for l in level], [])

    sort_idx = np.argsort(qs)
    quantiles = np.array(qs)[sort_idx]

    # Add default median
    quantiles = np.concatenate([np.array([50]), quantiles])
    quantiles = torch.Tensor(quantiles) / 100
    output_names = list(np.array(output_names)[sort_idx])
    output_names.insert(0, "-median")

    return quantiles, output_names


def quantiles_to_outputs(quantiles):
    output_names = []
    for q in quantiles:
        if q < 0.50:
            output_names.append(f"-lo-{np.round(100-200*q,2)}")
        elif q > 0.50:
            output_names.append(f"-hi-{np.round(100-200*(1-q),2)}")
        else:
            output_names.append("-median")
    return quantiles, output_names


def weighted_average(
    x: torch.Tensor, weights: Optional[torch.Tensor] = None, dim=None
) -> torch.Tensor:
    """
    Computes the weighted average of a given tensor across a given dim, masking
    values associated with weight zero,
    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.

    **Parameters:**<br>
    `x`: Input tensor, of which the average must be computed.<br>
    `weights`: Weights tensor, of the same shape as `x`.<br>
    `dim`: The dim along which to average `x`.<br>

    **Returns:**<br>
    `Tensor`: The tensor with values averaged along the specified `dim`.<br>
    """
    if weights is not None:
        weighted_tensor = torch.where(weights != 0, x * weights, torch.zeros_like(x))
        sum_weights = torch.clamp(
            weights.sum(dim=dim) if dim else weights.sum(), min=1.0
        )
        return (
            weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()
        ) / sum_weights
    else:
        return x.mean(dim=dim)

class RGM(torch.nn.Module):
    def __init__(
        self,
        n_components=1,
        level=[80, 90],
        quantiles=None,
        num_samples=1000,
        return_params=False,
        batch_correlation=False,
        horizon_correlation=False,
        weighted=False,
        posterior_regularization=False,
        posterior_weight=0.1,
    ):
        super(RGM, self).__init__()
        # Transform level to MQLoss parameters
        qs, self.output_names = level_to_outputs(level)
        qs = torch.Tensor(qs)

        # Transform quantiles to homogeneus output names
        if quantiles is not None:
            _, self.output_names = quantiles_to_outputs(quantiles)
            qs = torch.Tensor(quantiles)
        self.quantiles = torch.nn.Parameter(qs, requires_grad=False)
        self.num_samples = num_samples
        self.batch_correlation = batch_correlation
        self.horizon_correlation = horizon_correlation
        self.weighted = weighted
        
        # Posterior regularization parameters
        self.posterior_regularization = posterior_regularization
        self.posterior_weight = posterior_weight

        # If True, predict_step will return Distribution's parameters
        self.return_params = return_params

        mu_names = [f"-mu-{i}" for i in range(1, n_components + 1)]
        std_names = [f"-std-{i}" for i in range(1, n_components + 1)]
        if weighted:
            weight_names = [f"-weight-{i}" for i in range(1, n_components + 1)]
            self.param_names = [
                i for j in zip(mu_names, std_names, weight_names) for i in j
            ]
        else:
            self.param_names = [i for j in zip(mu_names, std_names) for i in j]

        if self.return_params:
            self.output_names = self.output_names + self.param_names

        # Add first output entry for the sample_mean
        self.output_names.insert(0, "")

        self.n_outputs = 2 + weighted
        self.n_components = n_components
        self.outputsize_multiplier = self.n_outputs * n_components
        self.is_distribution_output = True
        self.has_predicted = False

    def domain_map(self, output: torch.Tensor):
        output = output.reshape(
            output.shape[0], output.shape[1], -1, self.outputsize_multiplier
        )

        return torch.tensor_split(output, self.n_outputs, dim=-1)

    def scale_decouple(
        self,
        output,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        eps: float = 0.2,
    ):
        
        if self.weighted:
            means, stds, weights = output
            weights = F.softmax(weights, dim=-1)
        else:
            means, stds = output

        stds = F.softplus(stds)
        if (loc is not None) and (scale is not None):
            if loc.ndim == 3:
                loc = loc.unsqueeze(-1)
                scale = scale.unsqueeze(-1)
            means = (means * scale) + loc
            stds = (stds + eps) * scale

        if self.weighted:
            return (means, stds, weights)
        else:
            return (means, stds)

    def get_distribution(self, distr_args) -> Distribution:
        if self.weighted:
            means, stds, weights = distr_args
        else:
            means, stds = distr_args
            weights = torch.full_like(means, fill_value=1 / self.n_components)

        mix = Categorical(weights)
        components = Normal(loc=means, scale=stds)
        distr = MixtureSameFamily(
            mixture_distribution=mix, component_distribution=components
        )

        self.distr_mean = distr.mean

        return distr

    def sample(self, distr_args: torch.Tensor, num_samples: Optional[int] = None):
        if num_samples is None:
            num_samples = self.num_samples

        # Instantiate Scaled Decoupled Distribution
        distr = self.get_distribution(distr_args=distr_args)
        samples = distr.sample(sample_shape=(num_samples,))
        samples = samples.permute(
            1, 2, 3, 0
        )  # [samples, B, H, N] -> [B, H, N, samples]

        sample_mean = torch.mean(samples, dim=-1, keepdim=True)

        # Compute quantiles
        quantiles_device = self.quantiles.to(distr_args[0].device)
        quants = torch.quantile(input=samples, q=quantiles_device, dim=-1)
        quants = quants.permute(1, 2, 3, 0)  # [Q, B, H, N] -> [B, H, N, Q]

        return samples, sample_mean, quants

    def update_quantile(self, q: Optional[List[float]] = None):
        if q is not None:
            self.quantiles = nn.Parameter(
                torch.tensor(q, dtype=torch.float32), requires_grad=False
            )
            self.output_names = (
                [""]
                + [f"_ql{q_i}" for q_i in q]
                + self.return_params * self.param_names
            )
            self.has_predicted = True
        elif q is None and self.has_predicted:
            self.quantiles = nn.Parameter(
                torch.tensor([0.5], dtype=torch.float32), requires_grad=False
            )
            self.output_names = ["", "-median"] + self.return_params * self.param_names

    def __call__(
        self,
        y: torch.Tensor,
        distr_args: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
    ):
       
        # Instantiate Scaled Decoupled Distribution
        distr = self.get_distribution(distr_args=distr_args)
        x = distr._pad(y)
        log_prob_x = distr.component_distribution.log_prob(x)
        log_mix_prob = torch.log_softmax(distr.mixture_distribution.logits, dim=-1)
        
        # Store original log probabilities for posterior calculation
        original_log_prob_x = log_prob_x.clone()
        original_log_mix_prob = log_mix_prob.clone()
        
        if self.batch_correlation:
            log_prob_x = torch.sum(log_prob_x, dim=0, keepdim=True)
        if self.horizon_correlation:
            log_prob_x = torch.sum(log_prob_x, dim=1, keepdim=True)
        
        loss_values = -torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)
        
        # Add posterior entropy regularization
        if self.posterior_regularization and self.posterior_weight > 0:
            # Calculate posterior probabilities using original (non-summed) log probabilities
            log_posterior = original_log_prob_x + original_log_mix_prob
            log_posterior = log_posterior - torch.logsumexp(log_posterior, dim=-1, keepdim=True)
            
            # Calculate posterior entropy (negative entropy for minimization)
            posterior_entropy = -torch.sum(torch.exp(log_posterior) * log_posterior, dim=-1)
            
            # Apply same correlation operations as main loss if needed
            if self.batch_correlation:
                posterior_entropy = torch.sum(posterior_entropy, dim=0, keepdim=True)
            if self.horizon_correlation:
                posterior_entropy = torch.sum(posterior_entropy, dim=1, keepdim=True)
            
            # Add regularization term (positive weight minimizes entropy)
            loss_values = loss_values + self.posterior_weight * posterior_entropy

        return weighted_average(loss_values, weights=mask)