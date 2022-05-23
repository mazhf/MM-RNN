from util.ordered_easydict import OrderedEasyDict as edict
import os
from torch.nn import Conv2d, ConvTranspose2d
import numpy as np

cfg = edict()

# Multi_Scale_ConvLSTM  Multi_Modal_ConvLSTM
# Multi_Scale_TrajGRU  Multi_Modal_TrajGRU
# Multi_Scale_PredRNN  Multi_Modal_PredRNN
# Multi_Scale_PredRNN_plus2  Multi_Modal_PredRNN_plus2
# Multi_Scale_MIM  Multi_Modal_MIM
# Multi_Scale_MotionRNN_MIM  Multi_Modal_MotionRNN_MIM

cfg.model_name = 'Multi_Modal_ConvLSTM'
cfg.gpu = '0, 1, 2, 3'
cfg.gpu_nums = len(cfg.gpu.split(','))
cfg.work_path = 'MM-RNN'
cfg.data_path = 'Precipitation-Nowcasting'
cfg.dataset = 'MeteoNet'
cfg.height = 128
cfg.width = 128
cfg.TrajGRU_link_num = 10
cfg.lstm_hidden_state = 24
cfg.kernel_size = 3
cfg.batch = int(4 / len(cfg.gpu.split(',')))
cfg.LSTM_conv = Conv2d
cfg.LSTM_deconv = ConvTranspose2d
cfg.CONV_conv = Conv2d
cfg.in_len = 5
cfg.out_len = 15
if 'Multi_Modal' in cfg.model_name:
    cfg.train_epoch = 40
else:
    cfg.train_epoch = 20
cfg.valid_num = 10
cfg.valid_epoch = cfg.train_epoch // cfg.valid_num
cfg.LR = 0.0003
cfg.optimizer = 'Adam'
cfg.dataloader_thread = 0
cfg.data_type = np.float32
cfg.use_Attention = True
cfg.use_scheduled_sampling = False
cfg.use_loss_bw = False
cfg.LSTM_layers = 6
cfg.resume = False  # used only for test interrupt, optimizer config unknown for train interrupt.
cfg.resume_pth = ''

cfg.MODEL_LOG_SAVE_PATH = os.path.join('/home/mazhf', cfg.work_path, 'save', cfg.dataset, cfg.model_name)
cfg.DATASET_PATH = os.path.join('/home/mazhf', cfg.data_path, 'dataset', cfg.dataset, 'raw_radar_imgs')

cfg.HKO = edict()
cfg.HKO.THRESHOLDS = np.array([0.5, 2, 5, 10, 30])
cfg.HKO.CENTRAL_REGION = (120, 120, 360, 360)
cfg.HKO.BALANCING_WEIGHTS = (1, 1, 2, 5, 10, 30)

cfg.DWD = edict()
cfg.DWD.THRESHOLDS = np.array([0.5, 2, 5, 10, 30])
cfg.DWD.CENTRAL_REGION = (120, 120, 360, 360)
cfg.DWD.BALANCING_WEIGHTS = np.array([1, 1, 2, 5, 10, 30]).astype(np.float32)

cfg.MeteoNet = edict()
cfg.MeteoNet.THRESHOLDS = np.array([0.5, 2, 5, 10, 30])
cfg.MeteoNet.CENTRAL_REGION = (120, 120, 360, 360)
cfg.MeteoNet.BALANCING_WEIGHTS = (1, 1, 2, 5, 10, 30)

