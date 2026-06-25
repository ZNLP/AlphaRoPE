# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import is_torch_available, logging


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch


def _compute_default_rope_parameters(
    config: Optional[PretrainedConfig] = None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_default_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    elif config is not None:
        base = config.rope_theta
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))

    print("Using default RoPE")
    return inv_freq, attention_factor


def _compute_linear_scaling_rope_parameters(
    config: Optional[PretrainedConfig] = None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies with linear scaling. Credits to the Reddit user /u/kaiokendev
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_linear_scaling_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    if len(rope_kwargs) > 0:
        factor = rope_kwargs["factor"]
    elif config is not None:
        factor = config.rope_scaling["factor"]

    # Gets the default RoPE parameters
    inv_freq, attention_factor = _compute_default_rope_parameters(config, device, seq_len, **rope_kwargs)

    # Then applies linear scaling to the frequencies.
    # NOTE: originally, scaling was applied to the position_ids. However, we get `embs = inv_freq @ position_ids`, so
    # applying scaling to the inverse frequencies is equivalent.
    inv_freq /= factor

    print("Using Linear Scaling RoPE")
    return inv_freq, attention_factor

def _compute_ntk_parameters(
    config: Optional[PretrainedConfig] = None,
    device: Optional["torch.device"] = None,
    **rope_kwargs,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies with NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    # TODO (joao): use the new `original_max_position_embeddings` from rope_scaling
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_dynamic_ntk_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
        factor = rope_kwargs["factor"]
        scaling_factor = rope_kwargs["scaling_factor"]
    elif config is not None:
        base = config.rope_theta
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)
        factor = config.rope_scaling["factor"]
        scaling_factor = config.rope_scaling["scaling_factor"]

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    base = base * ((factor * scaling_factor) - (factor - 1)) ** (dim / (dim - 2))
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))

    print("Using NTK RoPE")
    return inv_freq, attention_factor

def _compute_dynamic_ntk_parameters(
    config: Optional[PretrainedConfig] = None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies with NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length, used to update the dynamic RoPE at inference time.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    # TODO (joao): use the new `original_max_position_embeddings` from rope_scaling
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_dynamic_ntk_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
        max_position_embeddings = rope_kwargs["max_position_embeddings"]
        factor = rope_kwargs["factor"]
    elif config is not None:
        base = config.rope_theta
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)
        max_position_embeddings = config.max_position_embeddings
        factor = config.rope_scaling["factor"]

    attention_factor = 1.0  # Unused in this type of RoPE

    # seq_len: default to max_position_embeddings, e.g. at init time
    seq_len = seq_len if seq_len is not None and seq_len > max_position_embeddings else max_position_embeddings
    print(seq_len)

    # Compute the inverse frequencies
    base = base * ((factor * seq_len / max_position_embeddings) - (factor - 1)) ** (dim / (dim - 2))
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))

    print("Using Dynamic NTK RoPE")
    return inv_freq, attention_factor


def _compute_yarn_parameters(
    config: PretrainedConfig, device: "torch.device", seq_len: Optional[int] = None, **rope_kwargs
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies with NTK scaling. Please refer to the
    [original paper](https://arxiv.org/abs/2309.00071)
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
    # No need to keep BC with yarn, unreleased when this new pattern was created.
    if len(rope_kwargs) > 0:
        raise ValueError(
            f"Unexpected arguments: `**rope_kwargs` should be unset in `_compute_yarn_parameters`, got {rope_kwargs}"
        )

    base = config.rope_theta
    partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    max_position_embeddings = config.max_position_embeddings
    factor = config.rope_scaling["factor"]

    # Sets the attention factor as suggested in the paper
    attention_factor = config.rope_scaling.get("attention_factor")
    if attention_factor is None:
        attention_factor = 0.1 * math.log(factor) + 1.0

    # Optional config options
    # beta_fast/beta_slow: as suggested in the paper, default to 32/1 (correspondingly)
    beta_fast = config.rope_scaling.get("beta_fast") or 32
    beta_slow = config.rope_scaling.get("beta_slow") or 1

    # Compute the inverse frequencies
    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        """Inverse dimension formula to find the dimension based on the number of rotations"""
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
        """Find dimension range bounds based on rotations"""
        low = math.floor(find_correction_dim(low_rot, dim, base, max_position_embeddings))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_position_embeddings))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001  # Prevent singularity

        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # Note on variable naming: "interpolation" comes from the original technique, where we interpolate the position IDs
    # to expand the possible context length. In other words, interpolation = apply scaling factor.
    pos_freqs = base ** (torch.arange(0, dim, 2).float().to(device) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    low, high = find_correction_range(beta_fast, beta_slow, dim, base, max_position_embeddings)

    # Get n-dimensional rotational scaling corrected for extrapolation
    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).float().to(device)
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )

    print("Using YaRN RoPE")

    return inv_freq, attention_factor

def _compute_mixed_radix_rope_parameters(
    config: PretrainedConfig, device: "torch.device", seq_len: Optional[int] = None, **rope_kwargs
) -> tuple["torch.Tensor", float]:
    """
    计算纯幂函数圈数缩放RoPE的逆频率。
    
    采用混合策略（参考 debug_power_circle.py）：
    - 低维部分 (j <= d0_half): 使用公式 λ(n) = factor^{(1/n)^p}
    - 高维部分 (j > d0_half): 直接使用 factor（恒为 factor）
    
    其中：
    - n = 训练时的旋转圈数 = L_train * ω_j / (2π)
    - factor = 扩展倍数
    - p = 幂指数（默认0.5）
    - d0_half = 临界维度（half_dim索引），用于区分低维和高维部分
    
    Args:
        config: 模型配置
        device: 设备
        seq_len: 序列长度（本方法未使用）
        rope_kwargs: 兼容性参数
        
    Returns:
        元组(逆频率张量, 注意力缩放因子)
    """
    if len(rope_kwargs) > 0:
        raise ValueError(f"不应传入rope_kwargs参数: {rope_kwargs}")

    # 基础配置提取
    base = config.rope_theta
    partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    max_position_embeddings = config.max_position_embeddings
    factor = config.rope_scaling["factor"]
    
    # 幂指数参数，默认1.0
    p = config.rope_scaling.get("power_exponent", 1.0)
    
    # 计算临界维度 d0
    d0_float = -0.5 * dim * math.log(2 * math.pi / max_position_embeddings) / math.log(base)
    d0 = int(2 * d0_float)
    print(f"d0:{d0}")
    
    # 初始化基础张量
    j = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (j / dim))
    
    # 计算每个维度在训练时的旋转圈数
    # n_j = L_train * ω_j / (2π)，其中ω_j = base^{-j}
    half_dim = dim // 2
    half_dim_indices = j / 2.0  # 将 j 转换为 half_dim 索引
    original_omega = base ** (-half_dim_indices / half_dim)
    rotations = max_position_embeddings * original_omega / (2 * math.pi)
    rotations = torch.clamp(rotations, min=1e-8)  # 避免除零
    
    # 计算高频和低频掩码（严格互斥）
    high_freq_mask = (j <= d0).float()
    low_freq_mask = (j > d0).float()
    
    # 分频段处理
    # 低维部分（j <= d0）：使用公式 λ = factor^{(1/rotations)^p}
    # 高维部分（j > d0）：直接使用 factor
    exponent = (1.0 / rotations) ** p  # (1/n)^p
    scaling_factors_high = factor ** exponent  # factor^{(1/n)^p}
    inv_freq_high = (inv_freq / scaling_factors_high) * high_freq_mask
    
    inv_freq_low = (inv_freq / factor) * low_freq_mask
    
    # 合并（安全加法）
    inv_freq = inv_freq_high + inv_freq_low
    
    # 纯形式不使用attention_factor缩放，始终返回1.0
    attention_factor = 1.0
    
    print("Using Mixed Radix RoPE")
    
    return inv_freq, attention_factor


# ====== 配置使用示例 ======
"""
"""

def _compute_ntk_yarn_hybrid_parameters(
    config: Optional[PretrainedConfig] = None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> tuple["torch.Tensor", float]:
    """
    Computes inverse frequencies with hybrid NTK-YaRN scaling:
    - High-freq (j <= d0): theta_j' = base^(-2j/d) * s^(-j/d0)
    - Low-freq (j > d0): theta_j' = base^(-2j/d) / s
    
    Args:
        config: Model configuration
        device: Target device
        seq_len: Current sequence length
        rope_kwargs: Compatibility parameters
        
    Returns:
        Tuple of (inv_freq, attention_factor)
    """
    # 1. Parameter initialization
    if config and rope_kwargs:
        raise ValueError("Cannot specify both config and rope_kwargs")
    
    if rope_kwargs:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
        max_position_embeddings = rope_kwargs.get("max_position_embeddings", 4096)
        factor = rope_kwargs["factor"]
    elif config:
        base = config.rope_theta
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)
        max_position_embeddings = config.max_position_embeddings
        factor = config.rope_scaling["factor"]

    attention_factor = config.rope_scaling.get("attention_factor")
    if attention_factor is None:
        attention_factor = 0.1 * math.log(factor) + 1.0
    
    # 2. Compute critical dimension d0
    d0 = -0.5 * dim * math.log(2 * math.pi / max_position_embeddings) / math.log(base)
    d0 = int(2*d0)
    print(f"d0:{d0}")
    
   # 3. Initialize base tensors
    j = torch.arange(0, dim, 2, dtype=torch.int64, device=device)
    inv_freq = 1.0 / (base ** (j / dim))
    
    


    
    # 1. 计算高频和低频掩码（严格互斥）
    high_freq_mask = (j <= d0).float()
    low_freq_mask = (j > d0).float()
    
    # 2. 分频段处理
    inv_freq_high = inv_freq * (factor**(-j / d0)) * high_freq_mask
    inv_freq_low = (inv_freq / factor) * low_freq_mask
    
    # 3. 合并（安全加法）
    inv_freq = inv_freq_high + inv_freq_low
        
        # # 验证临界点一致性
        # d0_idx = d0 // 2
        # if d0_idx < len(j):
        #     assert torch.allclose(
        #         inv_freq_high[d0_idx],
        #         inv_freq_low[d0_idx],
        #         atol=1e-6
        #     ), f"Critical dimension mismatch: {inv_freq_high[d0_idx]} vs {inv_freq_low[d0_idx]}"


    print("Using NTk-YaRN RoPE")
    return inv_freq, attention_factor


def _compute_my_new_parameters(
    config: Optional[PretrainedConfig] = None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> tuple["torch.Tensor", float]:
    """
    Computes inverse frequencies with hybrid NTK-YaRN scaling:
    - High-freq (j <= d0): theta_j' = base^(-2j/d) * s^(-j/d0)
    - Low-freq (j > d0): theta_j' = base^(-2j/d) / s
    
    Args:
        config: Model configuration
        device: Target device
        seq_len: Current sequence length
        rope_kwargs: Compatibility parameters
        
    Returns:
        Tuple of (inv_freq, attention_factor)
    """
    # 1. Parameter initialization
    if config and rope_kwargs:
        raise ValueError("Cannot specify both config and rope_kwargs")
    
    if rope_kwargs:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
        max_position_embeddings = rope_kwargs.get("max_position_embeddings")
        factor = rope_kwargs["factor"]
    elif config:
        base = config.rope_theta
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)
        max_position_embeddings = config.max_position_embeddings
        factor = config.rope_scaling["factor"]

    attention_factor = config.rope_scaling.get("attention_factor")
    if attention_factor is None:
        attention_factor = 0.1 * math.log(factor) + 1.0
    
    # 2. Compute critical dimension d0
    d0 = -0.5 * dim * math.log(2 * math.pi / (max_position_embeddings)) / math.log(base)
    d0 = int(2*d0)
    print(f"d0:{d0}")

    
   # 3. Initialize base tensors
    j = torch.arange(0, dim, 2, dtype=torch.int64, device=device)
    inv_freq = 1.0 / (base ** (j / dim))
    
    


    
    # 1. 计算高频和低频掩码（严格互斥）
    high_freq_mask = (j <= d0).float()
    low_freq_mask = (j > d0).float()
    
    alpha = 0.6*math.log(factor)
    print(f"alpha:{alpha}")
    # 2. 分频段处理
    inv_freq_high = inv_freq * (factor**(-(j / d0)**alpha)) * high_freq_mask
    inv_freq_low = (inv_freq / factor) * low_freq_mask


    
    # 3. 合并（安全加法）
    inv_freq = inv_freq_high + inv_freq_low
        


    print("Using My-new RoPE")
    return inv_freq, attention_factor





def _compute_longrope_parameters(
    config: PretrainedConfig, device: "torch.device", seq_len: Optional[int] = None, **rope_kwargs
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies with LongRoPE scaling. Please refer to the
    [original implementation](https://github.com/microsoft/LongRoPE)
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
    # TODO (joao): use the new `original_max_position_embeddings` from rope_scaling
    # No need to keep BC with longrope, unreleased when this new pattern was created.
    if len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` should be unset in `_compute_longrope_parameters`, got "
            f"{rope_kwargs}"
        )

    base = config.rope_theta
    partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    long_factor = config.rope_scaling["long_factor"]
    short_factor = config.rope_scaling["short_factor"]
    factor = config.rope_scaling.get("factor")
    attention_factor = config.rope_scaling.get("attention_factor")

    # NOTE: Phi3 (and potentially other models) modify `max_position_embeddings` and have a
    # `original_max_position_embeddings` field containing the pretrained value. They use the ratio between these two
    # values to compute the default attention scaling factor, instead of using `factor`.
    if hasattr(config, "original_max_position_embeddings"):
        original_max_position_embeddings = config.original_max_position_embeddings
        factor = config.max_position_embeddings / config.original_max_position_embeddings
    else:
        original_max_position_embeddings = config.max_position_embeddings

    # Sets the attention factor as suggested in the paper
    if attention_factor is None:
        if factor <= 1.0:
            attention_factor = 1.0
        else:
            attention_factor = math.sqrt(1 + math.log(factor) / math.log(original_max_position_embeddings))

    # Compute the inverse frequencies -- scaled based on the target sequence length
    if seq_len and seq_len > original_max_position_embeddings:
        ext_factors = torch.tensor(long_factor, dtype=torch.float32, device=device)
    else:
        ext_factors = torch.tensor(short_factor, dtype=torch.float32, device=device)
    inv_freq_shape = torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim
    inv_freq = 1.0 / (ext_factors * base**inv_freq_shape)
    print("Using LongRoPE")
    return inv_freq, attention_factor



def _compute_llama3_parameters(
    config: PretrainedConfig, device: "torch.device", seq_len: Optional[int] = None, **rope_kwargs
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies for llama 3.1.

    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
    # Gets the default RoPE parameters
    inv_freq, attention_factor = _compute_default_rope_parameters(config, device, seq_len, **rope_kwargs)

    factor = config.rope_scaling["factor"]  # `8` in the original implementation
    low_freq_factor = config.rope_scaling["low_freq_factor"]  # `1` in the original implementation
    high_freq_factor = config.rope_scaling["high_freq_factor"]  # `4` in the original implementation
    old_context_len = config.rope_scaling["original_max_position_embeddings"]  # `8192` in the original implementation

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    print("Using Llama3 RoPE")
    return inv_freq_llama, attention_factor


# def _compute_mixed_radix_rope_parameters(
#     config: Optional[PretrainedConfig] = None,
#     device: Optional["torch.device"] = None,
#     seq_len: Optional[int] = None,
#     **rope_kwargs,
# ) -> tuple["torch.Tensor", float]:
#     """
#     计算混合进制 RoPE 的逆频率参数。
#     通过将 β 进制转换为 β₁, β₂, ..., β_{d/2} 混合进制的方式扩展到 k 倍 Context。
#     其中 βₘ = β^λₘ，λ₁λ₂⋯λₘ = exp(a*m^b)，θₘ = n/(β^(m-1)*(λ₁λ₂⋯λₘ))
    
#     Args:
#         config: 模型配置
#         device: 目标设备
#         seq_len: 当前序列长度（未使用）
#         rope_kwargs: 兼容性参数
        
#     Returns:
#         Tuple of (inv_freq, attention_factor)
#     """
#     if config is not None and len(rope_kwargs) > 0:
#         raise ValueError(
#             "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
#             f"`_compute_mixed_radix_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
#         )
    
#     if len(rope_kwargs) > 0:
#         base = rope_kwargs["base"]
#         dim = rope_kwargs["dim"]
#         factor = rope_kwargs["factor"]
#         b = rope_kwargs.get("b", 1.0)  # 默认 b=1.0 对应 NTK-RoPE-fixed
#         alpha = rope_kwargs.get("alpha", 0.0)  # 高维增强参数，默认 0.0（原始公式）
#         gamma = rope_kwargs.get("gamma", 1.0)  # 高维增强幂次，默认 1.0
#     elif config is not None:
#         base = config.rope_theta
#         partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
#         head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
#         dim = int(head_dim * partial_rotary_factor)
#         factor = config.rope_scaling["factor"]
    
#     attention_factor = 1.0
    
#     # 根据公式 (10): β = 10000^(2/d) = base^(2/d)
#     d_half = dim // 2
#     m_indices = torch.arange(0, d_half, dtype=torch.float32, device=device)
    
#     # 新的约束条件（引入临界维度 d0）：
#     # λ₀λ₁⋯λ_{d0/2-1} = k
#     # λ₀λ₁⋯λ_{d/2-1} = 1.5k
    
#     # 计算临界维度 d0（和 YaRN 一样的方式）
#     max_position_embeddings = getattr(config, "max_position_embeddings", 4096) if config is not None else 4096
#     d0_float = -0.5 * dim * math.log(2 * math.pi / max_position_embeddings) / math.log(base)
#     d0 = int(2 * d0_float)  # 转换为索引，对应 j = 0, 2, 4, ..., d-2
#     d0_half = d0 // 2  # 转换为 half_dim 索引，对应 i = 0, 1, ..., d/2-1
#     d0_half = min(max(d0_half, 1), d_half - 1)  # 确保在有效范围内，至少为1（因为约束条件是 d0/2-1）
    
#     # 根据约束条件计算 b 和 a
#     # 使用改进的公式：λ₀λ₁⋯λᵢ = exp(a*((i+1)^b - 1))
#     # 这样可以减少低维的 scaling factor
#     # 约束1：λ₀λ₁⋯λ_{d0_half-1} = k，即 exp(a*((d0_half)^b - 1)) = k
#     # 约束2：λ₀λ₁⋯λ_{d_half-1} = 1.5k，即 exp(a*((d_half)^b - 1)) = 1.5k
#     # 从这两个方程可以解出：
#     # a*((d0_half)^b - 1) = ln(k)
#     # a*((d_half)^b - 1) = ln(1.5k)
#     # 因此：((d_half)^b - 1) / ((d0_half)^b - 1) = ln(1.5k) / ln(k)
#     # 这个方程需要数值求解 b，然后 a = ln(k) / ((d0_half)^b - 1)
#     log_k = math.log(factor)
#     log_1_5k = math.log(1.5 * factor)
    
#     # 使用 d0_half 和 d_half 来计算
#     if d0_half > 0 and d0_half < d_half:
#         # 数值求解 b：((d_half)^b - 1) / ((d0_half)^b - 1) = ln(1.5k) / ln(k)
#         target_ratio = log_1_5k / log_k
        
#         # 使用二分法求解 b
#         b_low, b_high = 0.1, 3.0
#         for _ in range(50):  # 最多迭代50次
#             b_mid = (b_low + b_high) / 2.0
#             half_power = d_half ** b_mid
#             d0_power = d0_half ** b_mid
#             ratio = (half_power - 1.0) / (d0_power - 1.0) if (d0_power - 1.0) > 1e-10 else float('inf')
            
#             if ratio < target_ratio:
#                 b_low = b_mid
#             else:
#                 b_high = b_mid
            
#             if abs(ratio - target_ratio) < 1e-6:
#                 break
        
#         b = (b_low + b_high) / 2.0
#     else:
#         # 如果 d0_half 无效，使用默认值
#         b = 1.0
    
#     # a = ln(k) / ((d0_half)^b - 1)
#     if d0_half > 0:
#         d0_power = d0_half ** b
#         denominator = d0_power - 1.0
#         if abs(denominator) > 1e-10:
#             a = log_k / denominator
#         else:
#             # 如果分母接近0，使用近似值
#             a = log_k / (d0_half ** b)
#     else:
#         a = log_k / (d_half ** b)
    
#     # 计算累积乘积 λ₀λ₁⋯λᵢ = exp(a*((i+1)^b - 1))
#     m_plus_one_power = (m_indices + 1.0) ** b
#     lambda_cumprod = torch.exp(a * (m_plus_one_power - 1.0))
    
#     # 计算 β^i
#     # 根据公式，β = base^(2/d)，所以 β^i = (base^(2/d))^i = base^(2*i/d)
#     beta_powers = base ** (2.0 * m_indices / dim)
    
#     # 根据修正后的公式：inv_freq[i] = 1/(β^i * (λ₀λ₁⋯λᵢ))
#     theta_base = beta_powers * lambda_cumprod
#     inv_freq = 1.0 / theta_base
    
#     # 验证约束条件
#     lambda_product_d0_minus_1 = lambda_cumprod[d0_half - 1].item() if d0_half > 0 else lambda_cumprod[0].item()
#     lambda_product_total = lambda_cumprod[-1].item()
    
#     # 输出计算出的参数
#     print(f"Using Mixed Radix RoPE (factor={factor}, computed b={b:.6f}, d0_half={d0_half})")
#     print(f"  Constraint check: λ₀...λ_{{d0/2-1}}={lambda_product_d0_minus_1:.6f} (expected {factor:.6f}), λ₀...λ_{{d/2-1}}={lambda_product_total:.6f} (expected {1.5*factor:.6f})")
    
#     return inv_freq, attention_factor


# This maps the "rope_type" string field in rope config to the corresponding function to compute the RoPE parameters
# from the model config. You can append new {'rope_type': callable} pairs to this dictionary to enable custom RoPE
# parameterizations, as long as the callable has the same signature.
ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
    "linear": _compute_linear_scaling_rope_parameters,
    "dynamic": _compute_dynamic_ntk_parameters,
    "ntk": _compute_ntk_parameters,
    "yarn": _compute_yarn_parameters,
    "ntk_yarn":_compute_ntk_yarn_hybrid_parameters,
    "my_new":_compute_my_new_parameters,
    "longrope": _compute_longrope_parameters,
    "llama3": _compute_llama3_parameters,
    "mixed_radix": _compute_mixed_radix_rope_parameters,
}


def _check_received_keys(
    rope_type: str,
    received_keys: set,
    required_keys: set,
    optional_keys: Optional[set] = None,
    ignore_keys: Optional[set] = None,
):
    """Compare the received keys in `config.rope_scaling` against the expected and optional keys"""
    # BC: "rope_type" was originally "type" -- let's check for "rope_type" when "type" is present
    if "type" in received_keys:
        received_keys -= {"type"}
        required_keys.add("rope_type")

    # Some models need to store model-specific keys, and we don't want to throw warning at them
    if ignore_keys is not None:
        received_keys -= ignore_keys

    missing_keys = required_keys - received_keys
    if missing_keys:
        raise KeyError(f"Missing required keys in `rope_scaling` for 'rope_type'='{rope_type}': {missing_keys}")

    if optional_keys is not None:
        unused_keys = received_keys - required_keys - optional_keys
    else:
        unused_keys = received_keys - required_keys
    if unused_keys:
        logger.warning(f"Unrecognized keys in `rope_scaling` for 'rope_type'='{rope_type}': {unused_keys}")


def _validate_default_rope_parameters(config: PretrainedConfig, ignore_keys: Optional[set] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
    required_keys = {"rope_type"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)


def _validate_linear_scaling_rope_parameters(config: PretrainedConfig, ignore_keys: Optional[set] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
    required_keys = {"rope_type", "factor"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

    factor = rope_scaling["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")


def _validate_dynamic_scaling_rope_parameters(config: PretrainedConfig, ignore_keys: Optional[set] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
    required_keys = {"rope_type", "factor"}
    # TODO (joao): update logic for the inclusion of `original_max_position_embeddings`
    optional_keys = {"original_max_position_embeddings"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

    factor = rope_scaling["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

def _validate_ntk_scaling_rope_parameters(config: PretrainedConfig, ignore_keys: Optional[set] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
    required_keys = {"rope_type", "factor", "scaling_factor"}
    # TODO (joao): update logic for the inclusion of `original_max_position_embeddings`
    optional_keys = {"original_max_position_embeddings"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

    factor = rope_scaling["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

    scaling_factor = rope_scaling["scaling_factor"]
    if scaling_factor is None or not isinstance(factor, float) or scaling_factor < 1.0:
        logger.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {scaling_factor}")


def _validate_yarn_parameters(config: PretrainedConfig, ignore_keys: Optional[set] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
    required_keys = {"rope_type", "factor"}
    optional_keys = {"attention_factor", "beta_fast", "beta_slow"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

    factor = rope_scaling["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

    attention_factor = rope_scaling.get("attention_factor")
    if attention_factor is not None and (not isinstance(attention_factor, float) or attention_factor < 0):
        logger.warning(
            f"`rope_scaling`'s attention_factor field must be a float greater than 0, got {attention_factor}"
        )
    beta_fast = rope_scaling.get("beta_fast")
    if beta_fast is not None and not isinstance(beta_fast, float):
        logger.warning(f"`rope_scaling`'s beta_fast field must be a float, got {beta_fast}")
    beta_slow = rope_scaling.get("beta_slow")
    if beta_slow is not None and not isinstance(beta_slow, float):
        logger.warning(f"`rope_scaling`'s beta_slow field must be a float, got {beta_slow}")

    if (beta_fast or 32) < (beta_slow or 1):
        logger.warning(
            f"`rope_scaling`'s beta_fast field must be greater than beta_slow, got beta_fast={beta_fast} "
            f"(defaults to 32 if None) and beta_slow={beta_slow} (defaults to 1 if None)"
        )

def _validate_ntk_yarn_parameters(config: PretrainedConfig, ignore_keys: Optional[set] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
    required_keys = {"rope_type", "factor"}
    optional_keys = {"attention_factor", "beta_fast", "beta_slow"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

    factor = rope_scaling["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

    attention_factor = rope_scaling.get("attention_factor")
    if attention_factor is not None and (not isinstance(attention_factor, float) or attention_factor < 0):
        logger.warning(
            f"`rope_scaling`'s attention_factor field must be a float greater than 0, got {attention_factor}"
        )
    beta_fast = rope_scaling.get("beta_fast")
    if beta_fast is not None and not isinstance(beta_fast, float):
        logger.warning(f"`rope_scaling`'s beta_fast field must be a float, got {beta_fast}")
    beta_slow = rope_scaling.get("beta_slow")
    if beta_slow is not None and not isinstance(beta_slow, float):
        logger.warning(f"`rope_scaling`'s beta_slow field must be a float, got {beta_slow}")

    if (beta_fast or 32) < (beta_slow or 1):
        logger.warning(
            f"`rope_scaling`'s beta_fast field must be greater than beta_slow, got beta_fast={beta_fast} "
            f"(defaults to 32 if None) and beta_slow={beta_slow} (defaults to 1 if None)"
        )

def _validate_my_new_parameters(config: PretrainedConfig, ignore_keys: Optional[set] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
    required_keys = {"rope_type", "factor"}
    optional_keys = {"attention_factor"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

    factor = rope_scaling["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

    attention_factor = rope_scaling.get("attention_factor")
    if attention_factor is not None and (not isinstance(attention_factor, float) or attention_factor < 0):
        logger.warning(
            f"`rope_scaling`'s attention_factor field must be a float greater than 0, got {attention_factor}"
        )




def _validate_longrope_parameters(config: PretrainedConfig, ignore_keys: Optional[set] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
    required_keys = {"rope_type", "short_factor", "long_factor"}
    # TODO (joao): update logic for the inclusion of `original_max_position_embeddings`
    optional_keys = {"attention_factor", "factor", "original_max_position_embeddings"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

    partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)

    short_factor = rope_scaling.get("short_factor")
    if not isinstance(short_factor, list) and all(isinstance(x, (int, float)) for x in short_factor):
        logger.warning(f"`rope_scaling`'s short_factor field must be a list of numbers, got {short_factor}")
    if not len(short_factor) == dim // 2:
        logger.warning(f"`rope_scaling`'s short_factor field must have length {dim // 2}, got {len(short_factor)}")

    long_factor = rope_scaling.get("long_factor")
    if not isinstance(long_factor, list) and all(isinstance(x, (int, float)) for x in long_factor):
        logger.warning(f"`rope_scaling`'s long_factor field must be a list of numbers, got {long_factor}")
    if not len(long_factor) == dim // 2:
        logger.warning(f"`rope_scaling`'s long_factor field must have length {dim // 2}, got {len(long_factor)}")

    # Handle Phi3 divergence: prefer the use of `attention_factor` and/or `factor` over
    # `original_max_position_embeddings` to compute internal variables. The latter lives outside `rope_scaling` and is
    # unique to longrope (= undesirable)
    if hasattr(config, "original_max_position_embeddings"):
        logger.warning_once(
            "This model has set a `original_max_position_embeddings` field, to be used together with "
            "`max_position_embeddings` to determine a scaling factor. Please set the `factor` field of `rope_scaling`"
            "with this ratio instead -- we recommend the use of this field over `original_max_position_embeddings`, "
            "as it is compatible with most model architectures."
        )
    else:
        factor = rope_scaling.get("factor")
        if factor is None:
            logger.warning("Missing required keys in `rope_scaling`: 'factor'")
        elif not isinstance(factor, float) or factor < 1.0:
            logger.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

        attention_factor = rope_scaling.get("attention_factor")
        if attention_factor is not None:
            if not isinstance(attention_factor, float) or attention_factor < 0.0:
                logger.warning(
                    f"`rope_scaling`'s attention_factor field must be a float greater than 0, got {attention_factor}"
                )


def _validate_llama3_parameters(config: PretrainedConfig, ignore_keys: Optional[set] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
    required_keys = {"rope_type", "factor", "original_max_position_embeddings", "low_freq_factor", "high_freq_factor"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

    factor = rope_scaling["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    if low_freq_factor is None or not isinstance(low_freq_factor, float):
        logger.warning(f"`rope_scaling`'s low_freq_factor field must be a float, got {low_freq_factor}")
    if high_freq_factor is None or not isinstance(high_freq_factor, float):
        logger.warning(f"`rope_scaling`'s high_freq_factor field must be a float, got {high_freq_factor}")
    if high_freq_factor <= low_freq_factor:
        logger.warning(
            "`rope_scaling`'s high_freq_factor field must be greater than low_freq_factor, got high_freq_factor="
            f"{high_freq_factor} and low_freq_factor={low_freq_factor}"
        )

    original_max_position_embeddings = rope_scaling["original_max_position_embeddings"]
    if original_max_position_embeddings is None or not isinstance(original_max_position_embeddings, int):
        logger.warning(
            "`rope_scaling`'s original_max_position_embeddings field must be an integer, got "
            f"{original_max_position_embeddings}"
        )
    if original_max_position_embeddings >= config.max_position_embeddings:
        logger.warning(
            "`rope_scaling`'s original_max_position_embeddings field must be less than max_position_embeddings, got "
            f"{original_max_position_embeddings} and max_position_embeddings={config.max_position_embeddings}"
        )


def _validate_mixed_radix_rope_parameters(config: PretrainedConfig, ignore_keys: Optional[set] = None):
    """

    """
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
    required_keys = {"rope_type", "factor"}
    optional_keys = {"attention_factor", "max_position_embeddings"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

    factor = rope_scaling["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

    attention_factor = rope_scaling.get("attention_factor")
    if attention_factor is not None and (not isinstance(attention_factor, float) or attention_factor < 0):
        logger.warning(
            f"`rope_scaling`'s attention_factor field must be a float greater than 0, got {attention_factor}"
        )


# Like `ROPE_INIT_FUNCTIONS`, this validation function mapping can be dynamically updated for custom RoPE types.
ROPE_VALIDATION_FUNCTIONS = {
    "default": _validate_default_rope_parameters,
    "linear": _validate_linear_scaling_rope_parameters,
    "dynamic": _validate_dynamic_scaling_rope_parameters,
    "ntk":_validate_ntk_scaling_rope_parameters,
    "yarn": _validate_yarn_parameters,
    "ntk_yarn":_validate_ntk_yarn_parameters,
    "my_new":_validate_my_new_parameters,
    "longrope": _validate_longrope_parameters,
    "llama3": _validate_llama3_parameters,
    "mixed_radix": _validate_mixed_radix_rope_parameters,
}


def rope_config_validation(config: PretrainedConfig, ignore_keys: Optional[set] = None):
    """
    Validate the RoPE config arguments, given a `PretrainedConfig` object
    """
    rope_scaling = getattr(config, "rope_scaling", None)  # not a default parameter in `PretrainedConfig`
    if rope_scaling is None:
        return

    # BC: "rope_type" was originally "type"
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
    validation_fn = ROPE_VALIDATION_FUNCTIONS.get(rope_type)
    if validation_fn is not None:
        validation_fn(config, ignore_keys=ignore_keys)
    else:
        logger.warning(
            f"Missing validation function mapping in `ROPE_VALIDATION_FUNCTIONS` for 'rope_type'='{rope_type}'"
        )