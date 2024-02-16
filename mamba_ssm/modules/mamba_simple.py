# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, selective_scan_ref

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None
causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None



@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t

@torch.no_grad()
def quantize_state_per_token_absmax(x, n_bits=8):
    # x: (bsize, nstate, seqlen), e.g.,[1, 16, 512]
    scales = x.abs().max(dim=1, keepdim=True)[0] # [1, 1, 512]
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    x.div_(scales).round_().mul_(scales)
    return x


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)


    def quant(self, act_scales=None):
        
        @torch.no_grad()
        def quantize_weight_per_tensor_absmax(w, n_bits=8):
            # w: (out_features, in_features)
            scales = w.abs().max()
            q_max = 2**(n_bits-1)-1
            scales.clamp_(min=1e-5).div_(q_max)
            w.div_(scales).round_().mul_(scales)
            return w

        @torch.no_grad()
        def smooth_fc(weight, act_scale, alpha=0.5):
            device = weight.device
            dtype = weight.dtype
            act_scale = act_scale.to(device).to(dtype)
            # linear fc weight shape [out_dim, in_dim]
            weight_scale = weight.abs().max(dim=0, keepdim=True)[0].clamp(min=1e-5) # [out_dim, in_dim] -> [1, in_dim]
            sm_scale = (act_scale[None, :].pow(alpha) / weight_scale.pow(1-alpha)).clamp(
                min=1e-5).to(device).to(dtype)
            return weight.mul_(sm_scale), sm_scale

        @torch.no_grad()
        def smooth_conv1d(weight, act_scale, alpha=0.5):
            device = weight.device
            dtype = weight.dtype
            act_scale = act_scale.to(device).to(dtype)
            # depth-wise conv1d weight shape [in_dim, 1, ksize]
            weight_scale = weight.abs().max(dim=2, keepdim=True)[0].clamp(min=1e-5) # [in_dim, 1, ksize] -> [in_dim, 1, 1]
            # act_scale[:, None, None] shape: [in_dim, 1, 1],  weight_scale shape: [in_dim, 1, 1]
            sm_scale = (act_scale[:, None, None].pow(alpha) / weight_scale.pow(1-alpha)).clamp(
                min=1e-5).to(device).to(dtype)
            return weight.mul_(sm_scale), torch.squeeze(sm_scale)[None, :, None]  # [in_dim] -> [1, in_dim, 1]
        
        """in_proj"""
        if act_scales is not None and "in_proj" in act_scales.keys():
            weight, self.in_proj_scale = smooth_fc(self.in_proj.weight, act_scales["in_proj"], alpha=0.5)
        else:
            weight = self.in_proj.weight
            self.in_proj_scale = torch.ones((1, weight.shape[1]), device=weight.device, dtype=weight.dtype)
        self.in_proj.weight = quantize_weight_per_tensor_absmax(weight)
        if self.in_proj.bias is not None:
            self.in_proj.bias = quantize_weight_per_tensor_absmax(self.in_proj.bias)

        """conv1d"""
        if act_scales is not None and "conv1d" in act_scales.keys():
            weight, self.conv1d_scale = smooth_conv1d(self.conv1d.weight, act_scales["conv1d"], alpha=0.5)
        else:
            weight = self.conv1d.weight
            self.conv1d_scale = torch.ones((1, weight.shape[0], 1), device=weight.device, dtype=weight.dtype)
        self.conv1d.weight = quantize_weight_per_tensor_absmax(weight)
        if self.conv1d.bias is not None:
            self.conv1d.bias = quantize_weight_per_tensor_absmax(self.conv1d.bias)

        """x_proj"""
        if act_scales is not None and "x_proj" in act_scales.keys():
            weight, self.x_proj_scale = smooth_fc(self.x_proj.weight, act_scales["x_proj"], alpha=0.5)
        else:
            weight = self.x_proj.weight
            self.x_proj_scale = torch.ones((1, weight.shape[1]), device=weight.device, dtype=weight.dtype)
        self.x_proj.weight = quantize_weight_per_tensor_absmax(weight)
        self.x_proj.weight = quantize_weight_per_tensor_absmax(self.x_proj.weight)

        """dt_proj"""
        if act_scales is not None and "dt_proj" in act_scales.keys():
            weight, self.dt_proj_scale = smooth_fc(self.dt_proj.weight, act_scales["dt_proj"], alpha=0.5)
        else:
            weight = self.dt_proj.weight
            self.dt_proj_scale = torch.ones((1, weight.shape[1]), device=weight.device, dtype=weight.dtype)
        self.dt_proj.weight = quantize_weight_per_tensor_absmax(weight)
        self.dt_proj.bias = quantize_weight_per_tensor_absmax(self.dt_proj.bias)

        """out_proj"""
        if act_scales is not None and "out_proj" in act_scales.keys():
            weight, self.out_proj_scale = smooth_fc(self.out_proj.weight, act_scales["out_proj"], alpha=0.5)
        else:
            weight = self.out_proj.weight
            self.out_proj_scale = torch.ones((1, weight.shape[1]), device=weight.device, dtype=weight.dtype)
        self.out_proj.weight = quantize_weight_per_tensor_absmax(weight)
        if self.out_proj.bias is not None:
            self.out_proj.bias = quantize_weight_per_tensor_absmax(self.out_proj.bias)
        
        """A_log"""
        self.A_log = quantize_weight_per_tensor_absmax(self.A_log)
        """D"""
        self.D = quantize_weight_per_tensor_absmax(self.D)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # hidden_states = quantize_activation_per_tensor_absmax(hidden_states, n_bits=8) # Naive
        hidden_states = quantize_activation_per_tensor_absmax(hidden_states.div(self.in_proj_scale), n_bits=8) # smooth quant hurts acc  0.76 -> 0.754
        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                # x = quantize_activation_per_tensor_absmax(x, n_bits=8)  # naive
                x = quantize_activation_per_tensor_absmax(x.div(self.conv1d_scale), n_bits=8) # smooth quant does not make much difference
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            # x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d) we divide into a few steps to apply quantization
            x_reshape = rearrange(x, "b d l -> (b l) d")
            # x_reshape = quantize_activation_per_tensor_absmax(x_reshape, n_bits=8) # naive
            x_reshape = quantize_activation_per_tensor_absmax(x_reshape.div(self.x_proj_scale), n_bits=8) # smooth quant !!! 0.729 -> 0.76 
            x_dbl = self.x_proj(x_reshape)  # (bl d)

            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            # dt = quantize_activation_per_tensor_absmax(dt, n_bits=8) # naive
            dt = quantize_activation_per_tensor_absmax(dt.div(self.dt_proj_scale), n_bits=8) # smooth quant!!! 0.756 -> 0.76
            dt = self.dt_proj.weight @ dt.t()
            
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

            assert self.activation in ["silu", "swish"]
            x = quantize_activation_per_tensor_absmax(x, n_bits=8) # naive
            dt = quantize_activation_per_tensor_absmax(dt, n_bits=8) # naive
            # A = quantize_activation_per_tensor_absmax(A.clamp_(min=-5), n_bits=8) # naive
            B = quantize_activation_per_tensor_absmax(B, n_bits=8) # naive
            C = quantize_activation_per_tensor_absmax(C, n_bits=8) # naive
            z = quantize_activation_per_tensor_absmax(z, n_bits=8) # naive
            # B = quantize_state_per_token_absmax(B, n_bits=8)
            # C = quantize_state_per_token_absmax(C, n_bits=8)
            # y = selective_scan_fn(
            y = selective_scan_ref(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            y = quantize_activation_per_tensor_absmax(y.div(self.out_proj_scale), n_bits=8) # smooth quant!!! 0.723 -> 0.76
            # y = quantize_activation_per_tensor_absmax(y, n_bits=8) # naive
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


if __name__ == "__main__":
    from functools import partial
    """
    MambaConfig(
        d_model=768, n_layer=24, vocab_size=50277,
        ssm_cfg={}, rms_norm=True, residual_in_fp32=True,
        fused_add_norm=True, pad_vocab_size_multiple=8
    )
    """
    device = "cuda"
    dtype = torch.float16
    norm_epsilon = 1e-5

    d_model = 768
    layer_idx = 0
    ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    rms_norm = True
    residual_in_fp32 = True
    fused_add_norm=True
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    print(block)

    # batch, seqlen, dim
    batch = 4
    seqlen = 8192
    dim = 768
    hidden_states = torch.rand((batch, seqlen, dim), device=device, dtype=dtype)
    residual = torch.rand((batch, seqlen, dim), device=device, dtype=dtype)
    # residual = None
    print(hidden_states.shape, residual.shape)
    hidden_states, residual = block(hidden_states, residual)
    print(hidden_states.shape, residual.shape)
