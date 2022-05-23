from models import *
from config import cfg
from collections import OrderedDict


b = cfg.batch
h = cfg.height
w = cfg.width
hs = cfg.lstm_hidden_state
if cfg.kernel_size == 5:
    k, s, p = 5, 1, 2
elif cfg.kernel_size == 3:
    k, s, p = 3, 1, 1
else:
    k, s, p = None, None, None

if cfg.model_name == 'Multi_Scale_ConvLSTM':
    rnn_param = Multi_Scale_ConvLSTM(input_channel=hs, output_channel=hs, b_h_w=(b, h, w), kernel_size=k, stride=s, padding=p)
elif cfg.model_name == 'Multi_Scale_TrajGRU':
    rnn_param = Multi_Scale_TrajGRU(input_channel=hs, output_channel=hs, b_h_w=(b, h, w), kernel_size=k, stride=s, padding=p)
elif cfg.model_name == 'Multi_Scale_PredRNN':
    rnn_param = Multi_Scale_PredRNN(input_channel=hs, output_channel=hs, b_h_w=(b, h, w), kernel_size=k, stride=s, padding=p)
elif cfg.model_name == 'Multi_Scale_PredRNN_plus2':
    rnn_param = Multi_Scale_PredRNN_plus2(input_channel=hs, output_channel=hs, b_h_w=(b, h, w), kernel_size=k, stride=s, padding=p)
elif cfg.model_name == 'Multi_Scale_MIM':
    rnn_param = Multi_Scale_MIM(input_channel=hs, output_channel=hs, b_h_w=(b, h, w), kernel_size=k, stride=s, padding=p)
elif cfg.model_name == 'Multi_Scale_MotionRNN_MIM':
    rnn_param = Multi_Scale_MotionRNN_MIM(input_channel=hs, output_channel=hs, b_h_w=(b, h, w), kernel_size=k, stride=s, padding=p)
elif cfg.model_name == 'Multi_Modal_ConvLSTM':
    rnn_param = Multi_Modal_ConvLSTM(input_channel=hs, output_channel=hs, b_h_w=(b, h, w), kernel_size=k, stride=s, padding=p)
elif cfg.model_name == 'Multi_Modal_TrajGRU':
    rnn_param = Multi_Modal_TrajGRU(input_channel=hs, output_channel=hs, b_h_w=(b, h, w), kernel_size=k, stride=s, padding=p)
elif cfg.model_name == 'Multi_Modal_PredRNN':
    rnn_param = Multi_Modal_PredRNN(input_channel=hs, output_channel=hs, b_h_w=(b, h, w), kernel_size=k, stride=s, padding=p)
elif cfg.model_name == 'Multi_Modal_PredRNN_plus2':
    rnn_param = Multi_Modal_PredRNN_plus2(input_channel=hs, output_channel=hs, b_h_w=(b, h, w), kernel_size=k, stride=s, padding=p)
elif cfg.model_name == 'Multi_Modal_MIM':
    rnn_param = Multi_Modal_MIM(input_channel=hs, output_channel=hs, b_h_w=(b, h, w), kernel_size=k, stride=s, padding=p)
elif cfg.model_name == 'Multi_Modal_MotionRNN_MIM':
    rnn_param = Multi_Modal_MotionRNN_MIM(input_channel=hs, output_channel=hs, b_h_w=(b, h, w), kernel_size=k, stride=s, padding=p)
else:
    rnn_param = None

if 'Multi_Scale' in cfg.model_name:
    params = [OrderedDict({'conv_embed': [1, hs, 1, 1, 0, 1]}),
              rnn_param,
              OrderedDict({'conv_fc': [hs, 1, 1, 1, 0, 1]})]
elif 'Multi_Modal' in cfg.model_name:
    params = [[OrderedDict({'conv_embed_radar': [1, hs, 1, 1, 0, 1]}), OrderedDict({'conv_embed_ele': [7, hs, 1, 1, 0, 1]})],
              rnn_param,
              [OrderedDict({'conv_fc_radar': [hs, 1, 1, 1, 0, 1]}), OrderedDict({'conv_fc_ele': [hs, 7, 1, 1, 0, 1]})]]

