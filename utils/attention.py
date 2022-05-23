from torch import nn
import torch

patch = 2


def make_patch(x):
    b, c, h, w = x.shape
    x = x.reshape(b, c * patch ** 2, h // patch, w // patch)
    return x


def patch_back(x):
    b, c, h, w = x.shape
    x = x.reshape(b, c // patch ** 2, h * patch, w * patch)
    return x


class MAM(nn.Module):
    def __init__(self, channel, H, W, layer):
        super().__init__()
        self.l = layer
        if self.l == 5:
            self.Query = nn.Conv2d(in_channels=channel * patch ** 2, out_channels=channel * patch ** 2, kernel_size=1,
                                   stride=1, padding=0)
            self.Key = nn.Conv2d(in_channels=channel * patch ** 2, out_channels=channel * patch ** 2, kernel_size=1,
                                 stride=1, padding=0)
            self.Value = nn.Conv2d(in_channels=channel * patch ** 2, out_channels=channel * patch ** 2, kernel_size=1,
                                   stride=1, padding=0)
        else:
            self.Query = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0)
            self.Key = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0)
            self.Value = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=-1)
        self.pos_embed_radar = nn.Parameter(
            torch.randn(1, channel * patch ** 2, H // patch, W // patch))  # position embedding broadcast
        self.pos_embed_modal = nn.Parameter(
            torch.randn(1, channel * patch ** 2, H // patch, W // patch))  # position embedding broadcast
        self.merge = nn.Conv2d(channel * 3, channel, 1, 1, 0)

    def forward(self, radar, modal):
        radar_raw = radar
        modal_raw = modal
        if self.l == 5:
            radar = make_patch(radar)
            radar = radar + self.pos_embed_radar
            modal = make_patch(modal)
            modal = modal + self.pos_embed_modal
        B, C, H, W = radar.shape
        Query = self.Query(radar).view(B, -1, H * W).permute(0, 2, 1)
        Key = self.Key(modal).view(B, -1, H * W)
        Similarity = torch.bmm(Query, Key)
        Similarity = self.softmax(Similarity)
        Value = self.Value(modal).view(B, -1, H * W)
        Attention = torch.bmm(Value, Similarity.permute(0, 2, 1))
        Attention = Attention.view(B, C, H, W)
        if self.l == 5:
            Attention = patch_back(Attention)
            radar = radar_raw
            modal = modal_raw
        out = Attention + radar + modal
        return out
