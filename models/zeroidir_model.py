
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

def compute_decom_lime(input_img):
    ill_map = torch.max(input_img, dim=1, keepdim=True)[0]
    ref_map = input_img / (ill_map + 1e-6)
    return ref_map, ill_map


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect"
            ),
            nn.SiLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect"
            )
        )

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):
        return self.model(x) + self.conv(x)


class DepthConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.depth_conv = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
            groups=in_ch
        )

        self.point_conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()

        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.se = nn.Sequential(
            nn.Conv2d(
                channel,
                channel // reduction,
                1,
                bias=False
            ),
            nn.ReLU(),
            nn.Conv2d(
                channel // reduction,
                channel,
                1,
                bias=False
            )
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)

        max_out = self.se(max_result)
        avg_out = self.se(avg_result)

        return self.sigmoid(max_out + avg_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="reflect"
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        avg_result = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        result = torch.cat(
            [max_result, avg_result],
            dim=1
        )

        return self.sigmoid(self.conv(result))


class CBAMBlock(nn.Module):
    def __init__(
        self,
        channel=512,
        reduction=16,
        kernel_size=5
    ):
        super().__init__()

        self.ca = ChannelAttention(
            channel=channel,
            reduction=reduction
        )

        self.sa = SpatialAttention(
            kernel_size=kernel_size
        )

    def forward(self, x):
        residual = x

        out = x * self.ca(x)
        out = out * self.sa(out)

        return out + residual


class RetinexDecom(nn.Module):
    def __init__(self, channels=64):
        super().__init__()

        self.conv0 = nn.Conv2d(
            2,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect"
        )

        self.blocks0 = nn.Sequential(
            ResBlock(channels, channels),
            ResBlock(channels, channels),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect"
            )
        )

        self.conv1 = nn.Conv2d(
            1,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect"
        )

        self.blocks1 = nn.Sequential(
            ResBlock(channels, channels),
            ResBlock(channels, channels),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect"
            )
        )

        self.channel_attention = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect"
            ),
            CBAMBlock(
                channel=channels,
                reduction=16,
                kernel_size=5
            ),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect"
            )
        )

        self.blocks2 = nn.Sequential(
            ResBlock(channels * 2, channels),
            ResBlock(channels, channels),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect"
            )
        )

        self.head_params = nn.Conv2d(
            channels,
            2,
            1
        )

        self.head_weights = nn.Conv2d(
            channels,
            2,
            1
        )

    @staticmethod
    def map_sigmoid(raw, lo, hi):
        return lo + (hi - lo) * torch.sigmoid(raw)

    def forward(self, input):
        init_reflectance, init_illumination = compute_decom_lime(input)

        structure_guidance = self.blocks0(
            self.conv0(
                torch.cat(
                    [
                        init_illumination,
                        init_reflectance.var(
                            1,
                            keepdim=True
                        ) + 1e-6
                    ],
                    dim=1
                )
            )
        )

        initial_gamma = self.blocks1(
            self.conv1(init_illumination)
        )

        gamma_map = self.channel_attention(
            self.blocks2(
                torch.cat(
                    (
                        initial_gamma,
                        structure_guidance
                    ),
                    dim=1
                )
            )
        )

        params = self.head_params(gamma_map)

        weights = F.softmax(
            self.head_weights(structure_guidance),
            dim=1
        )

        gamma_s = self.map_sigmoid(
            params[:, 0:1],
            0.0,
            1.0
        )

        gamma_m = self.map_sigmoid(
            params[:, 1:2],
            1.0,
            5.0
        )

        output_s = torch.pow(
            init_illumination,
            gamma_s
        )

        output_m = torch.pow(
            init_illumination,
            gamma_m
        )

        output_illumination = (
            weights[:, 0:1] * output_s +
            weights[:, 1:2] * output_m
        )

        output_illumination = torch.cat(
            [
                output_illumination,
                output_illumination,
                output_illumination
            ],
            dim=1
        )

        return output_illumination * init_reflectance


class Net(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.retinex = RetinexDecom(channels)

    def forward(self, x):
        low_img = x[:, :3, :, :]
        return self.retinex(low_img)

class Attend(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        scale = q.shape[-1] ** -0.5

        sim = torch.einsum(
            "b h i d, b h j d -> b h i j",
            q,
            k
        ) * scale

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        return torch.einsum(
            "b h i j, b h j d -> b h i d",
            attn,
            v
        )


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim, is_random=False):
        super().__init__()

        assert divisible_by(dim, 2)

        half_dim = dim // 2

        self.weights = nn.Parameter(
            torch.randn(half_dim),
            requires_grad=not is_random
        )

    def forward(self, x):
        x = rearrange(x, "b -> b 1")

        freqs = (
            x
            * rearrange(self.weights, "d -> 1 d")
            * 2
            * math.pi
        )

        fouriered = torch.cat(
            (freqs.sin(), freqs.cos()),
            dim=-1
        )

        return torch.cat(
            (x, fouriered),
            dim=-1
        )


class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        dim_head=32,
        num_mem_kv=4
    ):
        super().__init__()

        self.scale = dim_head ** -0.5
        self.heads = heads

        hidden_dim = heads * dim_head

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(
            torch.randn(
                2,
                heads,
                dim_head,
                num_mem_kv
            )
        )

        self.to_qkv = nn.Conv2d(
            dim,
            hidden_dim * 3,
            1,
            bias=False
        )

        self.to_out = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                dim,
                1
            ),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)

        q, k, v = map(
            lambda t: rearrange(
                t,
                "b (heads c) h w -> b heads c (h w)",
                heads=self.heads
            ),
            qkv
        )

        mem_k, mem_v = self.mem_kv

        mem_k = repeat(
            mem_k,
            "heads c n -> b heads c n",
            b=b
        )

        mem_v = repeat(
            mem_v,
            "heads c n -> b heads c n",
            b=b
        )

        k = torch.cat(
            (mem_k, k),
            dim=-1
        )

        v = torch.cat(
            (mem_v, v),
            dim=-1
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum(
            "b h d n, b h e n -> b h d e",
            k,
            v
        )

        out = torch.einsum(
            "b h d e, b h d n -> b h e n",
            context,
            q
        )

        out = rearrange(
            out,
            "b h c (x y) -> b (h c) x y",
            h=self.heads,
            x=h,
            y=w
        )

        return self.to_out(out)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        dim_head=32,
        num_mem_kv=4
    ):
        super().__init__()

        self.heads = heads

        hidden_dim = heads * dim_head

        self.norm = RMSNorm(dim)

        self.attend = Attend()

        self.mem_kv = nn.Parameter(
            torch.randn(
                2,
                heads,
                num_mem_kv,
                dim_head
            )
        )

        self.to_qkv = nn.Conv2d(
            dim,
            hidden_dim * 3,
            1,
            bias=False
        )

        self.to_out = nn.Conv2d(
            hidden_dim,
            dim,
            1
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(
            3,
            dim=1
        )

        q, k, v = map(
            lambda t: rearrange(
                t,
                "b (heads c) h w -> b heads (h w) c",
                heads=self.heads
            ),
            qkv
        )

        mem_k, mem_v = self.mem_kv

        mem_k = repeat(
            mem_k,
            "heads n d -> b heads n d",
            b=b
        )

        mem_v = repeat(
            mem_v,
            "heads n d -> b heads n d",
            b=b
        )

        k = torch.cat(
            (mem_k, k),
            dim=-2
        )

        v = torch.cat(
            (mem_v, v),
            dim=-2
        )

        out = self.attend(
            q,
            k,
            v
        )

        out = rearrange(
            out,
            "b heads (h w) d -> b (heads d) h w",
            h=h,
            w=w
        )

        return self.to_out(out)


class Stage2Upsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()

        dim_out = default(
            dim_out,
            dim
        )

        self.net = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode="nearest"
            ),
            nn.Conv2d(
                dim,
                dim_out,
                3,
                padding=1
            )
        )

    def forward(self, x):
        return self.net(x)


class Stage2Downsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()

        dim_out = default(
            dim_out,
            dim
        )

        self.net = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (c p1 p2) h w",
                p1=2,
                p2=2
            ),
            nn.Conv2d(
                dim * 4,
                dim_out,
                1
            )
        )

    def forward(self, x):
        return self.net(x)


class Stage2Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()

        self.proj = nn.Conv2d(
            dim,
            dim_out,
            3,
            padding=1
        )

        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class Stage2ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        time_emb_dim=None
    ):
        super().__init__()

        self.mlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    time_emb_dim,
                    dim_out * 2
                )
            )
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Stage2Block(
            dim,
            dim_out
        )

        self.block2 = Stage2Block(
            dim_out,
            dim_out
        )

        self.res_conv = (
            nn.Conv2d(
                dim,
                dim_out,
                1
            )
            if dim != dim_out
            else nn.Identity()
        )

    def forward(self, x, time_emb=None):
        scale_shift = None

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)

            time_emb = rearrange(
                time_emb,
                "b c -> b c 1 1"
            )

            scale_shift = time_emb.chunk(
                2,
                dim=1
            )

        h = self.block1(
            x,
            scale_shift=scale_shift
        )

        h = self.block2(h)

        return h + self.res_conv(x)


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        sinusoidal_pos_emb_theta=10000,
        attn_dim_head=32,
        attn_heads=4,
        full_attn=None,
        flash_attn=False
    ):
        super().__init__()

        self.channels = channels

        input_channels = channels * 2

        init_dim = default(
            init_dim,
            dim
        )

        self.init_conv = nn.Conv2d(
            input_channels,
            init_dim,
            7,
            padding=3
        )

        dims = [
            init_dim,
            *map(
                lambda m: dim * m,
                dim_mults
            )
        ]

        in_out = list(
            zip(
                dims[:-1],
                dims[1:]
            )
        )

        time_dim = dim * 4

        if learned_sinusoidal_cond or random_fourier_features:
            pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim,
                random_fourier_features
            )
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            pos_emb = SinusoidalPosEmb(
                dim,
                theta=sinusoidal_pos_emb_theta
            )
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            pos_emb,
            nn.Linear(
                fourier_dim,
                time_dim
            ),
            nn.GELU(),
            nn.Linear(
                time_dim,
                time_dim
            )
        )

        if full_attn is None:
            full_attn = (
                *((False,) * (len(dim_mults) - 1)),
                True
            )

        full_attn = cast_tuple(
            full_attn,
            len(dim_mults)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), use_full_attn) in enumerate(
            zip(
                in_out,
                full_attn
            )
        ):
            is_last = ind >= (
                num_resolutions - 1
            )

            attention_class = (
                Attention
                if use_full_attn
                else LinearAttention
            )

            self.downs.append(
                nn.ModuleList([
                    Stage2ResnetBlock(
                        dim_in,
                        dim_in,
                        time_emb_dim=time_dim
                    ),
                    Stage2ResnetBlock(
                        dim_in,
                        dim_in,
                        time_emb_dim=time_dim
                    ),
                    attention_class(
                        dim_in,
                        dim_head=attn_dim_head,
                        heads=attn_heads
                    ),
                    Stage2Downsample(
                        dim_in,
                        dim_out
                    )
                    if not is_last
                    else nn.Conv2d(
                        dim_in,
                        dim_out,
                        3,
                        padding=1
                    )
                ])
            )

        mid_dim = dims[-1]

        self.mid_block1 = Stage2ResnetBlock(
            mid_dim,
            mid_dim,
            time_emb_dim=time_dim
        )

        self.mid_attn = Attention(
            mid_dim,
            heads=attn_heads,
            dim_head=attn_dim_head
        )

        self.mid_block2 = Stage2ResnetBlock(
            mid_dim,
            mid_dim,
            time_emb_dim=time_dim
        )

        for ind, ((dim_in, dim_out), use_full_attn) in enumerate(
            zip(
                reversed(in_out),
                reversed(full_attn)
            )
        ):
            is_last = ind >= (
                num_resolutions - 1
            )

            attention_class = (
                Attention
                if use_full_attn
                else LinearAttention
            )

            self.ups.append(
                nn.ModuleList([
                    Stage2ResnetBlock(
                        dim_out + dim_in,
                        dim_out,
                        time_emb_dim=time_dim
                    ),
                    Stage2ResnetBlock(
                        dim_out + dim_in,
                        dim_out,
                        time_emb_dim=time_dim
                    ),
                    attention_class(
                        dim_out,
                        dim_head=attn_dim_head,
                        heads=attn_heads
                    ),
                    Stage2Upsample(
                        dim_out,
                        dim_in
                    )
                    if not is_last
                    else nn.Conv2d(
                        dim_out,
                        dim_in,
                        3,
                        padding=1
                    )
                ])
            )

        self.out_dim = default(
            out_dim,
            channels * (
                2
                if learned_variance
                else 1
            )
        )

        self.final_res_block = Stage2ResnetBlock(
            init_dim * 2,
            init_dim,
            time_emb_dim=time_dim
        )

        self.final_conv = nn.Conv2d(
            init_dim,
            self.out_dim,
            1
        )

    @property
    def downsample_factor(self):
        return 2 ** (
            len(self.downs) - 1
        )

    def forward(self, x, time, x_cond):
        assert all(
            divisible_by(
                d,
                self.downsample_factor
            )
            for d in x.shape[-2:]
        )

        x = torch.cat(
            (
                x_cond,
                x
            ),
            dim=1
        )

        x = self.init_conv(x)

        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:

            x = block1(
                x,
                t
            )

            h.append(x)

            x = block2(
                x,
                t
            )

            x = attn(x) + x

            h.append(x)

            x = downsample(x)

        x = self.mid_block1(
            x,
            t
        )

        x = self.mid_attn(x) + x

        x = self.mid_block2(
            x,
            t
        )

        for block1, block2, attn, upsample in self.ups:

            x = torch.cat(
                (
                    x,
                    h.pop()
                ),
                dim=1
            )

            x = block1(
                x,
                t
            )

            x = torch.cat(
                (
                    x,
                    h.pop()
                ),
                dim=1
            )

            x = block2(
                x,
                t
            )

            x = attn(x) + x

            x = upsample(x)

        x = torch.cat(
            (
                x,
                r
            ),
            dim=1
        )

        x = self.final_res_block(
            x,
            t
        )

        return self.final_conv(x)

from einops.layers.torch import Rearrange


def exists(x):
    return x is not None


def default(val, d):
    return val if exists(val) else (d() if callable(d) else d)


def cast_tuple(t, length=1):
    if isinstance(t, tuple):
        return t
    return (t,) * length


def divisible_by(numer, denom):
    return numer % denom == 0


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * self.scale


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2

        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(
                half_dim,
                device=device
            ) * -emb
        )

        emb = x[:, None] * emb[None, :]

        return torch.cat(
            (emb.sin(), emb.cos()),
            dim=-1
        )

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(
        b,
        *((1,) * (len(x_shape) - 1))
    )


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02

    return torch.linspace(
        beta_start,
        beta_end,
        timesteps,
        dtype=torch.float64
    )


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1

    t = torch.linspace(
        0,
        timesteps,
        steps,
        dtype=torch.float64
    ) / timesteps

    alphas_cumprod = torch.cos(
        (t + s) / (1 + s) * math.pi * 0.5
    ) ** 2

    alphas_cumprod = (
        alphas_cumprod /
        alphas_cumprod[0]
    )

    betas = 1 - (
        alphas_cumprod[1:] /
        alphas_cumprod[:-1]
    )

    return torch.clip(
        betas,
        0,
        0.999
    )


def sigmoid_beta_schedule(
    timesteps,
    start=-3,
    end=3,
    tau=1,
    clamp_min=1e-5
):
    steps = timesteps + 1

    t = torch.linspace(
        0,
        timesteps,
        steps,
        dtype=torch.float64
    ) / timesteps

    v_start = torch.tensor(
        start / tau
    ).sigmoid()

    v_end = torch.tensor(
        end / tau
    ).sigmoid()

    alphas_cumprod = (
        -(
            (t * (end - start) + start) / tau
        ).sigmoid()
        + v_end
    ) / (v_end - v_start)

    alphas_cumprod = (
        alphas_cumprod /
        alphas_cumprod[0]
    )

    betas = 1 - (
        alphas_cumprod[1:] /
        alphas_cumprod[:-1]
    )

    return torch.clip(
        betas,
        0,
        0.999
    )


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        timesteps=1000,
        sampling_timesteps=20,
        beta_schedule="sigmoid",
        ddim_sampling_eta=0.0
    ):
        super().__init__()

        self.model = model
        self.channels = model.channels

        self.stage1_Net = Net()
        self.stage1_Net.eval()

        for param in self.stage1_Net.parameters():
            param.requires_grad = False

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps)
        else:
            raise ValueError(
                f"Unknown beta schedule: {beta_schedule}"
            )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(
            alphas,
            dim=0
        )

        alphas_cumprod_prev = F.pad(
            alphas_cumprod[:-1],
            (1, 0),
            value=1.0
        )

        self.num_timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        self.register_buffer(
            "betas",
            betas.float()
        )

        self.register_buffer(
            "alphas_cumprod",
            alphas_cumprod.float()
        )

        self.register_buffer(
            "alphas_cumprod_prev",
            alphas_cumprod_prev.float()
        )

        self.register_buffer(
            "sqrt_alphas_cumprod",
            torch.sqrt(alphas_cumprod).float()
        )

        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(
                1.0 - alphas_cumprod
            ).float()
        )

        self.register_buffer(
            "sqrt_recip_alphas_cumprod",
            torch.sqrt(
                1.0 / alphas_cumprod
            ).float()
        )

        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(
                1.0 / alphas_cumprod - 1
            ).float()
        )

    def q_sample(
        self,
        x_start,
        t,
        noise=None
    ):
        if noise is None:
            noise = torch.randn_like(x_start)

        return (
            extract(
                self.sqrt_alphas_cumprod,
                t,
                x_start.shape
            ) * x_start
            +
            extract(
                self.sqrt_one_minus_alphas_cumprod,
                t,
                x_start.shape
            ) * noise
        )

    def predict_start_from_noise(
        self,
        x_t,
        t,
        noise
    ):
        return (
            extract(
                self.sqrt_recip_alphas_cumprod,
                t,
                x_t.shape
            ) * x_t
            -
            extract(
                self.sqrt_recipm1_alphas_cumprod,
                t,
                x_t.shape
            ) * noise
        )

    def predict_noise_from_start(
        self,
        x_t,
        t,
        x_start
    ):
        return (
            extract(
                self.sqrt_recip_alphas_cumprod,
                t,
                x_t.shape
            ) * x_t
            -
            x_start
        ) / extract(
            self.sqrt_recipm1_alphas_cumprod,
            t,
            x_t.shape
        )

    def model_predictions(
        self,
        x,
        t,
        x_cond
    ):
        pred_x_start = self.model(
            x,
            t,
            x_cond
        )

        pred_x_start = pred_x_start.clamp(
            -1.0,
            1.0
        )

        pred_noise = self.predict_noise_from_start(
            x,
            t,
            pred_x_start
        )

        return pred_noise, pred_x_start

    @torch.inference_mode()
    def ddim_sample(
        self,
        x_cond,
        return_all_timesteps=False
    ):
        batch, _, h, w = x_cond.shape

        device = x_cond.device

        times = torch.linspace(
            -1,
            self.num_timesteps - 1,
            steps=self.sampling_timesteps + 1,
            device=device
        )

        times = list(
            reversed(
                times.long().tolist()
            )
        )

        time_pairs = list(
            zip(
                times[:-1],
                times[1:]
            )
        )

        img = torch.randn(
            batch,
            self.channels,
            h,
            w,
            device=device
        )

        images = [img]

        for time, time_next in time_pairs:

            time_cond = torch.full(
                (batch,),
                time,
                device=device,
                dtype=torch.long
            )

            pred_noise, pred_x_start = self.model_predictions(
                img,
                time_cond,
                x_cond
            )

            if time_next < 0:
                img = pred_x_start
                images.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = (
                self.ddim_sampling_eta
                * torch.sqrt(
                    (1 - alpha / alpha_next)
                    * (1 - alpha_next)
                    / (1 - alpha)
                )
            )

            c = torch.sqrt(
                1 - alpha_next - sigma ** 2
            )

            noise = torch.randn_like(img)

            img = (
                pred_x_start * torch.sqrt(alpha_next)
                +
                c * pred_noise
                +
                sigma * noise
            )

            images.append(img)

        if return_all_timesteps:
            return torch.stack(
                images,
                dim=1
            )

        return img


class ZeroIDIRModel(nn.Module):
    def __init__(
        self,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        timesteps=1000,
        sampling_timesteps=20
    ):
        super().__init__()

        self.stage1 = Net()

        self.unet = Unet(
            dim=dim,
            dim_mults=dim_mults,
            channels=channels
        )

        self.diffusion = GaussianDiffusion(
            self.unet,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps
        )

        self.diffusion.stage1_Net = self.stage1

    def forward(self, x, time):
        stage1_output = self.stage1(x)

        return self.unet(
            x,
            time,
            stage1_output * 2 - 1
        )
