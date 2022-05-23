from torch import nn
import torch
from config import cfg
import numpy as np
from util.utils import make_layers


def scheduled_sampling(shape, eta):
    S, B, C, H, W = shape
    # 随机种子已固定, 生成[0,1)随机数，形状 = (pre_len-1行，batch_size列)
    random_flip = np.random.random_sample((S - 1, B))  # outS-1 * B
    true_token = (random_flip < eta)  # 若eta为1，true_token[t, i]全部为True，mask元素全为1
    one = torch.FloatTensor(1, C, H, W).fill_(1.0).cuda()  # 1*C*H*W
    zero = torch.FloatTensor(1, C, H, W).fill_(0.0).cuda()  # 1*C*H*W
    masks = []
    for t in range(S - 1):
        masks_b = []  # B*C*H*W
        for i in range(B):
            if true_token[t, i]:
                masks_b.append(one)
            else:
                masks_b.append(zero)
        mask = torch.cat(masks_b, 0)  # along batch size
        masks.append(mask)  # outS-1 * B*C*H*W
    return masks


class Model(nn.Module):
    def __init__(self, embed, rnn, fc):
        super().__init__()
        if 'Multi_Scale' in cfg.model_name:
            self.embed = make_layers(embed)
            self.embed_ele = None
            self.fc = make_layers(fc)
            self.fc_ele = None
        elif 'Multi_Modal' in cfg.model_name:
            self.embed = make_layers(embed[0])
            self.embed_ele = make_layers(embed[1])
            self.fc = make_layers(fc[0])
            self.fc_ele = make_layers(fc[1])
        self.rnns = rnn
        self.in_len = cfg.in_len
        self.out_len = cfg.out_len
        self.use_ss = cfg.use_scheduled_sampling

    def ss_ele(self, truth, pred, mask, t):
        mix_lis = []
        for e in range(7):
            truth_ele = truth[:, e, ...]  # b 1 h w
            pred_ele = pred[:, e, ...]  # b 1 h w
            mix = mask[t - self.in_len] * truth_ele + (1 - mask[t - self.in_len]) * pred_ele  # b 1 h w
            mix_lis.append(mix)  # 7 b 1 h w
        mix_lis = torch.stack(mix_lis)  # 7 b 1 h w
        mix_lis = torch.squeeze(mix_lis, dim=2)  # 7 b h w
        mix_lis = mix_lis.permute(1, 0, 2, 3)  # b 7 h w
        return mix_lis

    def forward(self, inputs):
        x, eta = inputs  # x: sb8hw
        shape = [cfg.out_len, cfg.batch, 1, cfg.height, cfg.width]  # out_s b 1 h w
        if self.use_ss:
            mask = scheduled_sampling(shape, eta)  # out_s-1 b 1 h w
        else:
            mask = None
        outputs = []
        layer_hiddens = None  # every batch init with zeros: same with Lin et al. SA-ConvLSTM
        output = None
        if 'Multi_Scale' in cfg.model_name:
            m = None
        elif 'Multi_Modal' in cfg.model_name:
            m = [None, None]
        for t in range(self.in_len + self.out_len - 1):
            if t < self.in_len:
                input = x[t]  # b8hw
            else:
                if self.use_ss:  # ss only with radar
                    input_radar = mask[t - self.in_len] * torch.unsqueeze(x[t, :, 0, ...], dim=1) + (1 - mask[t - self.in_len]) * torch.unsqueeze(output[:, 0, ...], dim=1)  # b1hw
                    input_ele = self.ss_ele(x[t, :, 1:, ...], output[:, 1:, ...], mask, t)  # b7hw
                    input = torch.cat([input_radar, input_ele], dim=1)  # b8hw
                else:
                    input = output  # b8hw
            output, m, layer_hiddens = self.rnns(input, m, layer_hiddens, self.embed, self.embed_ele, self.fc, self.fc_ele)
            outputs.append(output)
        outputs = torch.stack(outputs)  # s-1b8hw
        return outputs




