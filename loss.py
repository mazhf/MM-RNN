from torch import nn
from config import cfg
import torch


class Weighted_mse_mae(nn.Module):
    def __init__(self, radar_weight=1.0, ele_weight=1.0):
        super().__init__()
        self.radar_weight = radar_weight
        self.ele_weight = ele_weight

    def forward(self, truth_all, pred_all):  # s-1 b 8 h w
        # radar
        truth = truth_all[:, :, 0, ...]  # s-1 b h w
        truth = torch.unsqueeze(truth, dim=2)  # s-1 b 1 h w
        pred = pred_all[:, :, 0, ...]  # s-1 b h w
        pred = torch.unsqueeze(pred, dim=2)  # s-1 b 1 h w
        differ = truth - pred  # s-1 b 1 h w
        mse = torch.sum(differ ** 2, (2, 3, 4))  # s b
        mae = torch.sum(torch.abs(differ), (2, 3, 4))  # s b
        mse = torch.mean(mse)
        mae = torch.mean(mae)
        loss = mse + mae

        # element
        if 'Multi_Modal' in cfg.model_name:
            truth_ele = truth_all[:, :, 1:, ...]  # s-1 b 7 h w
            pred_ele = pred_all[:, :, 1:, ...]  # s-1 b 7 h w
            differ_ele = truth_ele - pred_ele  # s-1 b 7 h w
            mse_ele = torch.sum(differ_ele ** 2, (2, 3, 4))  # s b
            mae_ele = torch.sum(torch.abs(differ_ele), (2, 3, 4))  # s b
            mse_ele = torch.mean(mse_ele)
            mae_ele = torch.mean(mae_ele)
            loss_ele = mse_ele + mae_ele
            loss_all = self.radar_weight * loss + self.ele_weight * loss_ele
        else:
            loss_all = loss
        return loss_all
