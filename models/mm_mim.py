from torch import nn
import torch
import sys
sys.path.append("..")
from config import cfg
from models.ms_mim import MIM_cell
from util.attention import MAM


class Multi_Modal_MIM(nn.Module):
    def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding):
        super().__init__()
        self.n_layers = cfg.LSTM_layers
        B, H, W = b_h_w
        lstm = [MIM_cell(input_channel, output_channel, [B, H, W], kernel_size, stride, padding),
                MIM_cell(input_channel, output_channel, [B, H // 2, W // 2], kernel_size, stride, padding),
                MIM_cell(input_channel, output_channel, [B, H // 4, W // 4], kernel_size, stride, padding),
                MIM_cell(input_channel, output_channel, [B, H // 4, W // 4], kernel_size, stride, padding),
                MIM_cell(input_channel, output_channel, [B, H // 2, W // 2], kernel_size, stride, padding),
                MIM_cell(input_channel, output_channel, [B, H, W], kernel_size, stride, padding)]
        self.lstm = nn.ModuleList(lstm)

        lstm_ele = [MIM_cell(input_channel, output_channel, [B, H, W], kernel_size, stride, padding),
                    MIM_cell(input_channel, output_channel, [B, H // 2, W // 2], kernel_size, stride, padding),
                    MIM_cell(input_channel, output_channel, [B, H // 4, W // 4], kernel_size, stride, padding),
                    MIM_cell(input_channel, output_channel, [B, H // 4, W // 4], kernel_size, stride, padding),
                    MIM_cell(input_channel, output_channel, [B, H // 2, W // 2], kernel_size, stride, padding),
                    MIM_cell(input_channel, output_channel, [B, H, W], kernel_size, stride, padding)]
        self.lstm_ele = nn.ModuleList(lstm_ele)

        self.downs = nn.ModuleList([nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2)])
        self.downs_ele = nn.ModuleList([nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2)])

        self.ups = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear'), nn.Upsample(scale_factor=2, mode='bilinear')])
        self.ups_ele = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear'), nn.Upsample(scale_factor=2, mode='bilinear')])

        self.downs_m = nn.ModuleList([nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2)])
        self.downs_m_ele = nn.ModuleList([nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2)])

        self.ups_m = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear'), nn.Upsample(scale_factor=2, mode='bilinear')])
        self.ups_m_ele = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear'), nn.Upsample(scale_factor=2, mode='bilinear')])

        self.downs_xt_1 = nn.ModuleList([nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2)])
        self.downs_xt_1_ele = nn.ModuleList([nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2)])

        self.ups_xt_1 = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear'), nn.Upsample(scale_factor=2, mode='bilinear')])
        self.ups_xt_1_ele = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear'), nn.Upsample(scale_factor=2, mode='bilinear')])

        if cfg.use_Attention:
            Att = [MAM(input_channel, H, W, 5), MAM(input_channel, H // 2, W // 2, 4), MAM(input_channel, H // 4, W // 4, 3)]
            self.Att = nn.ModuleList(Att)

        print('This is Multi Modal MIM!')

    def forward(self, x_all, m_all, layer_hiddens, embed, embed_ele, fc, fc_ele):  # b8hw
        x = x_all[:, 0, ...]  # bhw
        x = torch.unsqueeze(x, dim=1)  # b1hw
        x_ele = x_all[:, 1:, ...]  # b7hw

        x = embed(x)  # bchw
        x_ele = embed_ele(x_ele)  # bchw

        m, m_ele = m_all

        next_layer_hiddens = []
        out = []
        out_ele = []
        for l in range(self.n_layers):
            if layer_hiddens is not None:
                hiddens = layer_hiddens[l][0]
                xt_1 = layer_hiddens[l - 1][0][0]
                hiddens_ele = layer_hiddens[l][1]
                xt_1_ele = layer_hiddens[l - 1][1][0]
                if l == 0:
                    xt_1 = None
                    xt_1_ele = None
                elif l == 1:
                    xt_1 = self.downs_xt_1[0](xt_1)
                    xt_1_ele = self.downs_xt_1_ele[0](xt_1_ele)
                elif l == 2:
                    xt_1 = self.downs_xt_1[1](xt_1)
                    xt_1_ele = self.downs_xt_1_ele[1](xt_1_ele)
                elif l == 4:
                    xt_1 = self.ups_xt_1[0](xt_1)
                    xt_1_ele = self.ups_xt_1_ele[0](xt_1_ele)
                elif l == 5:
                    xt_1 = self.ups_xt_1[1](xt_1)
                    xt_1_ele = self.ups_xt_1_ele[1](xt_1_ele)
            else:
                hiddens = None
                hiddens_ele = None
                xt_1 = None
                xt_1_ele = None
            # elements
            x_ele, m_ele, next_hiddens_ele = self.lstm_ele[l](x_ele, xt_1_ele, m_ele, hiddens_ele, l)
            out_ele.append(x_ele)

            # radar elements fuse
            if cfg.use_Attention:
                if l == 3:
                    x = self.Att[2](x, x_ele)
                elif l == 4:
                    x = self.Att[1](x, x_ele)
                elif l == 5:
                    x = self.Att[0](x, x_ele)
            else:
                if l == 3:
                    x = x + x_ele
                elif l == 4:
                    x = x + x_ele
                elif l == 5:
                    x =x + x_ele

            # radar
            x, m, next_hiddens = self.lstm[l](x, xt_1, m, hiddens, l)
            out.append(x)

            if l == 0:
                x = self.downs[0](x)
                x_ele = self.downs_ele[0](x_ele)
                m = self.downs_m[0](m)
                m_ele = self.downs_m_ele[0](m_ele)
            elif l == 1:
                x = self.downs[1](x)
                x_ele = self.downs_ele[1](x_ele)
                m = self.downs_m[1](m)
                m_ele = self.downs_m_ele[1](m_ele)
            elif l == 3:
                x = self.ups[1](x) + out[1]
                x_ele = self.ups_ele[1](x_ele) + out_ele[1]
                m = self.ups_m[1](m)
                m_ele = self.ups_m_ele[1](m_ele)
            elif l == 4:
                x = self.ups[0](x) + out[0]
                x_ele = self.ups_ele[0](x_ele) + out_ele[0]
                m = self.ups_m[0](m)
                m_ele = self.ups_m_ele[0](m_ele)

            next_layer_hiddens.append([next_hiddens, next_hiddens_ele])

        x = fc(x)  # b1hw
        x_ele = fc_ele(x_ele)  # b7hw
        return torch.cat([x, x_ele], dim=1), [m, m_ele], next_layer_hiddens