import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

"""using ConvNext https://github.com/facebookresearch/ConvNeXt"""


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7,
                                padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class NoGoNet(nn.Module):
    """
    ????????????[B,2,9,9]
    ????????????([B,9,9],[B])?????????????????????9*9?????????????????????v
    """

    def __init__(self, scale=16):
        super().__init__()
        self.preproccess = nn.Sequential(
            nn.Conv2d(2, scale*4, kernel_size=3, padding=1),
            nn.LayerNorm((9, 9)),
            nn.GELU(),
        )
        blks = []
        for _ in range(scale):
            blks.append(Block(scale*4))
        self.blks = nn.Sequential(*blks)
        self.policy_head = nn.Sequential(
            nn.Conv2d(scale*4, 2, kernel_size=1),
            nn.LayerNorm((9, 9)),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(9*9*2, 9*9),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(scale*4, 1, kernel_size=1),
            nn.Flatten(),
            nn.LayerNorm((9*9*1)),
            nn.GELU(),
            nn.Linear(9*9*1, scale*4),
            nn.GELU(),
            nn.Linear(scale*4, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        n = x.shape[0]
        x = self.preproccess(x)
        x = self.blks(x)
        a = self.policy_head(x)
        a = torch.reshape(a, (n, 81))
        a = F.softmax(a, 1).reshape((n, 9, 9))
        b = self.value_head(x)
        return a, b


if __name__ == "__main__":
    a = torch.ones((1, 2, 9, 9))
    m = NoGoNet(scale=4)
    p, v = m(a)
    torch.save(m, "models/test.pt")
    print(p)
    print(v)
