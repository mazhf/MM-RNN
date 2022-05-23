from torch import nn
import torch
import sys
sys.path.append("..")
from config import cfg


class PredRNN_plus2_cell(nn.Module):
    def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding):
        super().__init__()
        self._batch_size, self._state_height, self._state_width = b_h_w
        self._conv_x_h_c = cfg.LSTM_conv(in_channels=input_channel * 3, out_channels=output_channel * 3,
                                         kernel_size=kernel_size, stride=stride, padding=padding)
        self._conv_x_c_m = cfg.LSTM_conv(in_channels=input_channel * 3, out_channels=output_channel * 3,
                                         kernel_size=kernel_size, stride=stride, padding=padding)
        self._conv_m = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel,
                                     kernel_size=1, stride=1, padding=0)
        self._conv_o = cfg.LSTM_conv(in_channels=input_channel * 3, out_channels=output_channel,
                                     kernel_size=kernel_size, stride=stride, padding=padding)
        self._conv_c_m = cfg.LSTM_conv(in_channels=2 * input_channel, out_channels=output_channel,
                                       kernel_size=1, stride=1, padding=0)
        self._conv_x = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel * 2,
                                     kernel_size=kernel_size, stride=stride, padding=padding)
        self._conv_z = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel * 2,
                                     kernel_size=kernel_size, stride=stride, padding=padding)

        self._input_channel = input_channel
        self._output_channel = output_channel

    def forward(self, x, m, hiddens, l):
        if hiddens is None:
            c = torch.zeros((x.shape[0], self._input_channel, self._state_height, self._state_width),
                            dtype=torch.float).cuda()
            h = torch.zeros((x.shape[0], self._input_channel, self._state_height, self._state_width),
                            dtype=torch.float).cuda()
            z = torch.zeros((x.shape[0], self._input_channel, self._state_height, self._state_width),
                            dtype=torch.float).cuda()
        else:
            h, c, z = hiddens
        if m is None:
            m = torch.zeros((x.shape[0], self._output_channel, self._state_height, self._state_width),
                            dtype=torch.float).cuda()
        # GHU
        if l == 1:
            x2h = self._conv_x(x)
            z2h = self._conv_z(z)
            p, s = torch.chunk((x2h + z2h), 2, dim=1)
            p = torch.tanh(p)
            s = torch.sigmoid(s)
            next_z = s * p + (1 - s) * z
            x = next_z
        else:
            next_z = None

        # Causal LSTM
        x_h_c = torch.cat([x, h, c], dim=1)
        i, f, g = torch.chunk(self._conv_x_h_c(x_h_c), 3, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        next_c = f * c + i * g

        x_c_m = torch.cat([x, next_c, m], dim=1)
        i_m, f_m, g_m = torch.chunk(self._conv_x_c_m(x_c_m), 3, dim=1)
        i_m = torch.sigmoid(i_m)
        f_m = torch.sigmoid(f_m)
        g_m = torch.tanh(g_m)
        next_m = f_m * torch.tanh(self._conv_m(m)) + i_m * g_m

        o = torch.tanh(self._conv_o(torch.cat([x, next_c, next_m], dim=1)))
        next_h = o * torch.tanh(self._conv_c_m(torch.cat([next_c, next_m], dim=1)))

        ouput = next_h
        next_hiddens = [next_h, next_c, next_z]
        return ouput, next_m, next_hiddens


class Multi_Scale_PredRNN_plus2(nn.Module):
    def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding):
        super().__init__()
        self.n_layers = cfg.LSTM_layers
        B, H, W = b_h_w
        lstm = [PredRNN_plus2_cell(input_channel, output_channel, [B, H, W], kernel_size, stride, padding),
                PredRNN_plus2_cell(input_channel, output_channel, [B, H // 2, W // 2], kernel_size, stride, padding),
                PredRNN_plus2_cell(input_channel, output_channel, [B, H // 4, W // 4], kernel_size, stride, padding),
                PredRNN_plus2_cell(input_channel, output_channel, [B, H // 4, W // 4], kernel_size, stride, padding),
                PredRNN_plus2_cell(input_channel, output_channel, [B, H // 2, W // 2], kernel_size, stride, padding),
                PredRNN_plus2_cell(input_channel, output_channel, [B, H, W], kernel_size, stride, padding)]
        self.lstm = nn.ModuleList(lstm)

        self.downs = nn.ModuleList([nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2)])
        self.ups = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear'), nn.Upsample(scale_factor=2, mode='bilinear')])

        self.downs_m = nn.ModuleList([nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2)])
        self.ups_m = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear'), nn.Upsample(scale_factor=2, mode='bilinear')])

        print('This is Multi Scale PredRNN++!')

    # [radar, dd, ff, hu, td, t, psl, elevation]
    def forward(self, x_all, m, layer_hiddens, embed, embed_ele, fc, fc_ele):  # b8hw
        x = x_all[:, 0, ...]  # bhw
        x = torch.unsqueeze(x, dim=1)  # b1hw
        x = embed(x)
        next_layer_hiddens = []
        out = []
        for l in range(self.n_layers):
            if layer_hiddens is not None:
                hiddens = layer_hiddens[l]
            else:
                hiddens = None
            x, m, next_hiddens = self.lstm[l](x, m, hiddens, l)
            out.append(x)
            if l == 0:
                x = self.downs[0](x)
                m = self.downs_m[0](m)
            elif l == 1:
                x = self.downs[1](x)
                m = self.downs_m[1](m)
            elif l == 3:
                x = self.ups[1](x) + out[1]
                m = self.ups_m[1](m)
            elif l == 4:
                x = self.ups[0](x) + out[0]
                m = self.ups_m[0](m)
            next_layer_hiddens.append(next_hiddens)
        x = fc(x)  # b1hw
        x_ele = torch.zeros([cfg.batch, 7, cfg.height, cfg.width]).to(x.device)  # b7hw
        return torch.cat([x, x_ele], dim=1), m, next_layer_hiddens
