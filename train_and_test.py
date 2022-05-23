import torch
from config import cfg
import numpy as np
from util.evaluation import Evaluation
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os
import shutil
import pandas as pd
import time
from util.visualization import save_movie, save_image
import cv2


IN_LEN = cfg.in_len
OUT_LEN = cfg.out_len
gpu_nums = cfg.gpu_nums
dd = {'max': 0.0, 'min': 360.0}  # wrong should be 0-360
ff = {'max': 37.54846851792292, 'min': 0.0}  # wrong should be > 0
hu = {'max': 100, 'min': 0.0}  # wrong should be 0-100
td = {'max': 313.9070046005785, 'min': 221.48327260054808}
psl = {'max': 104626.48964329725, 'min': 95867.01941430081}
t = {'max': 311.937838308806, 'min': 241.35155875346243}


def normal_ele_cuda(x):
    x = torch.cuda.FloatTensor(x)
    x = x.expand(IN_LEN + OUT_LEN, cfg.batch, 1, x.shape[0], x.shape[1])  # s x b x 1 x h x w
    x = x / 255.0
    return x


def normalize_data_cuda(batch):  # b * 7 * s * 1 * h * w
    batch = torch.squeeze(batch, dim=3)   # b * 7 * s * h * w
    batch = batch.permute(2, 0, 1, 3, 4)  # S x B x 7 x H x W
    # ['radar', 'dd', 'ff', 'hu', 'td', 't', 'psl']
    batch[:, :, 0, ...] = batch[:, :, 0, ...] / 255.0
    batch[:, :, 1, ...] = (batch[:, :, 1, ...] - dd['min']) / (dd['max'] - dd['min'])
    batch[:, :, 2, ...] = (batch[:, :, 2, ...] - ff['min']) / (ff['max'] - ff['min'])
    batch[:, :, 3, ...] = (batch[:, :, 3, ...] - hu['min']) / (hu['max'] - hu['min'])
    batch[:, :, 4, ...] = (batch[:, :, 4, ...] - td['min']) / (td['max'] - td['min'])
    batch[:, :, 5, ...] = (batch[:, :, 5, ...] - t['min']) / (t['max'] - t['min'])
    batch[:, :, 6, ...] = (batch[:, :, 6, ...] - psl['min']) / (psl['max'] - psl['min'])
    batch = batch.type(torch.float32)
    return batch.cuda()


def reduce_tensor(tensor):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= gpu_nums
    return rt


# is main process ?
def is_master_proc(gpu_nums=gpu_nums):
    return torch.distributed.get_rank() % gpu_nums == 0


def count_params(params_lis, model):
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0
    for param in model.parameters():
        mulValue = param.numel()
        Total_params += mulValue
        if param.requires_grad:
            Trainable_params += mulValue
        else:
            NonTrainable_params += mulValue
    params_lis.append(Total_params)
    params_lis.append(Trainable_params)
    params_lis.append(NonTrainable_params)
    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')
    return params_lis


def train_and_test(model, optimizer, criterion, start_epoch, train_epoch, valid_epoch, save_checkpoint_epoch, loader, train_sampler):
    elev_path = os.path.join(os.path.dirname(cfg.DATASET_PATH), 'NW_elevation.png')
    elevation = cv2.imread(elev_path, flags=0)
    elevation = cv2.resize(elevation, (cfg.width, cfg.height))
    elevation = normal_ele_cuda(elevation)
    """
    只在主进程创建，删除，写入文件。loss和度量是all_reduce后的结果，同步到每个进程，这样主进程中的就是均值
    """
    train_valid_metrics_save_path, model_save_path, writer, save_path, test_metrics_save_path = [None] * 5
    train_loader, test_loader, valid_loader = loader
    start = time.time()
    eval_ = Evaluation(seq_len=IN_LEN + OUT_LEN - 1, use_central=False)
    save_path = cfg.MODEL_LOG_SAVE_PATH
    model_save_path = os.path.join(save_path, 'models')
    log_save_path = os.path.join(save_path, 'logs')
    test_metrics_save_path = os.path.join(save_path, "test_metrics.xlsx")
    if is_master_proc():
        writer = SummaryWriter(log_save_path)
    if not cfg.resume and is_master_proc():
        # 初始化保存路径，覆盖前面训练的models, logs, metrics
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path)
        os.makedirs(model_save_path)
        os.makedirs(log_save_path)
    # 训练、验证
    train_loss = 0.0
    valid_times = 0
    params_lis = []
    eta = 1.0
    delta = 1 / (train_epoch * len(train_loader))
    if cfg.resume and is_master_proc():
        params_lis = count_params(params_lis, model)
    for epoch in range(start_epoch + 1, train_epoch + 1):
        if is_master_proc():
            print('epoch: ', epoch)
        pbar = tqdm(total=len(train_loader), desc="train_batch", disable=not is_master_proc())  # 进度条
        # train
        train_sampler.set_epoch(epoch)  # 防止每个epoch被分配到每块卡上的数据都一样，虽然数据已经平分给各个卡，但不设置的话，每次平分的数据都一样
        for idx, train_batch in enumerate(train_loader, 1):
            train_batch = normalize_data_cuda(train_batch)
            train_batch = torch.cat([train_batch, elevation], dim=2)  # S x B x 8 x H x W
            train_batch_bw = torch.flip(train_batch, dims=[0])
            model.train()
            # fw
            optimizer.zero_grad()
            train_pred = model([train_batch, eta])
            loss = criterion(train_batch[1:, ...], train_pred)
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=5)
            # bw
            if cfg.use_loss_bw:
                optimizer.zero_grad()
                train_pred_bw = model([train_batch_bw, eta])
                loss_bw = criterion(train_batch_bw[1:, ...], train_pred_bw)
                loss_bw.backward()
                optimizer.step()
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=5)
                # sum
                loss = (loss + loss_bw) / 2

            # 更新lr、loss、metrics
            loss = reduce_tensor(loss)  # all reduce
            train_loss += loss.item()
            eta -= delta
            eta = max(eta, 0)
            pbar.update(1)

            # 计算参数量
            if epoch == 1 and idx == 1 and is_master_proc():
                params_lis = count_params(params_lis, model)
        pbar.close()

        # valid
        if epoch % valid_epoch == 0:
            valid_times += 1
            train_loss = train_loss / (len(train_loader) * valid_epoch)
            with torch.no_grad():
                model.eval()
                valid_loss = 0.0
                for valid_batch in valid_loader:
                    valid_batch = normalize_data_cuda(valid_batch)
                    valid_batch = torch.cat([valid_batch, elevation], dim=2)  # S x B x 8 x H x W
                    valid_pred = model([valid_batch, eta])
                    loss = criterion(valid_batch[1:, ...], valid_pred)
                    loss = reduce_tensor(loss)  # all reduce
                    valid_loss += loss.item()
                valid_loss = valid_loss / len(valid_loader)
            # 第一个参数可以简单理解为保存tensorboard中图的名称，第二个参数是可以理解为Y轴数据，第三个参数可以理解为X轴数据。
            if is_master_proc():
                writer.add_scalars("loss", {"train": train_loss, "valid": valid_loss}, epoch)  # plot loss
            train_loss = 0.0

        # save model
        if is_master_proc() and epoch % save_checkpoint_epoch == 0:
            torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))

    # test
    if is_master_proc():
        writer.close()
        eval_.clear_all()
        with torch.no_grad():
            model.eval()
            test_loss = 0.0
            test_times = 0
            for test_batch in test_loader:
                test_times += 1
                test_batch = normalize_data_cuda(test_batch)
                test_batch = torch.cat([test_batch, elevation], dim=2)  # S x B x 8 x H x W
                test_pred = model([test_batch, 0])
                loss = criterion(test_batch[1:, ...], test_pred)
                test_loss += loss.item()
                test_batch_numpy = test_batch.cpu().numpy()
                test_batch_numpy = test_batch_numpy[1:, :, 0, ...]  # s-1bhw
                test_batch_numpy = test_batch_numpy[:, :, np.newaxis, ...]  # s-1b1hw
                test_pred_numpy = np.clip(test_pred.detach().cpu().numpy(), 0.0, 1.0)
                test_pred_numpy = test_pred_numpy[:, :, 0, ...]  # s-1bhw
                test_pred_numpy = test_pred_numpy[:, :, np.newaxis, ...]  # s-1b1hw
                eval_.update(test_batch_numpy, test_pred_numpy)  # s-1b1hw only radar
            test_metrics_lis = eval_.get_metrics()
            test_loss = test_loss / test_times
            test_metrics_lis.append(test_loss)
            end = time.time()
            running_time = round((end - start) / 3600, 2)
            print("===============================")
            print('Running time: {} hours'.format(running_time))
            print("===============================")
            save_test_metrics(test_metrics_lis, test_metrics_save_path, params_lis, running_time)
            eval_.clear_all()
        test_demo(test_loader, model, elevation)


def nan_to_num(metrics):
    for i in range(len(metrics)):
        metrics[i] = np.nan_to_num(metrics[i])
    return metrics


def save_test_metrics(m_lis, path, p_lis, run_tim):
    m_lis = nan_to_num(m_lis)
    col0 = ['test_ssim', 'test_psnr', 'test_gdl', 'test_balanced_mse', 'test_balanced_mae', 'test_mse', 'test_mae',
            'test_pod_0.5', 'test_pod_2', 'test_pod_5', 'test_pod_10', 'test_pod_30',
            'test_far_0.5', 'test_far_2', 'test_far_5', 'test_far_10', 'test_far_30',
            'test_csi_0.5', 'test_csi_2', 'test_csi_5', 'test_csi_10', 'test_csi_30',
            'test_hss_0.5', 'test_hss_2', 'test_hss_5', 'test_hss_10', 'test_hss_30',
            'test_loss', 'Total_params', 'Trainable_params', 'NonTrainable_params', 'time']
    add_col0 = [str(i) for i in range(1, IN_LEN + OUT_LEN)]
    col1 = []
    add_col1 = []
    for i in range(len(m_lis)):
        metric = m_lis[i]
        if i in [7, 8, 9, 10]:
            for j in range(len(cfg.HKO.THRESHOLDS)):
                col1.append(metric[:, j].mean())
                if (i in [9, 10]) and (j == len(cfg.HKO.THRESHOLDS) - 1):
                    add_col1.append(metric[:, j])
        elif i == 11:
            col1.append(metric)
        else:
            col1.append(metric.mean())
    col1 += p_lis
    col1.append(run_tim)
    df = pd.DataFrame()
    df['0'] = col0
    df['1'] = col1
    df.columns = ['Metrics', 'Value']
    df.to_excel(path, index=0)
    add_df = pd.DataFrame()
    add_df['0'] = add_col0
    add_df['1'] = add_col1[0]
    add_df['2'] = add_col1[1]
    add_df.columns = ['frame', 'csi', 'hss']
    split = path.split('.')
    add_path = split[0] + '_framewise_csi30_hss30.' + split[1]
    add_df.to_excel(add_path, index=0)


def test_demo(test_loader, model, elevation):
    channel_names = ['radar', 'dd', 'ff', 'hu', 'td', 't', 'psl', 'elevation']
    for i in range(len(test_loader)):
        test_batch = list(test_loader)[i]
        test_batch = normalize_data_cuda(test_batch)
        test_batch = torch.cat([test_batch, elevation], dim=2)  # S x B x 8 x H x W
        input = test_batch
        with torch.no_grad():
            output = model([input, 0])
        output = np.clip(output.cpu().numpy(), 0.0, 1.0)
        input = input[1:, 0, :, :, :]  # s-18hw
        input = input.cpu().numpy()
        output = output[:, 0, :, :, :]  # s-18hw
        for m in range(len(channel_names)):
            in_out_ = []
            for j in range(input.shape[0]):
                in_out_elem = np.concatenate((input[j, m, ...], output[j, m, ...]), axis=1)
                in_out_.append(in_out_elem)
            in_out_ = np.array(in_out_)
            test_demo_save_path = os.path.join(cfg.MODEL_LOG_SAVE_PATH, 'demo', 'random_seed_' + str(i + 1) + '_demo', channel_names[m])
            if not os.path.exists(test_demo_save_path):
                os.makedirs(test_demo_save_path)
            in_ = input[:, m, ...]  # s-1hw
            out_ = output[:, m, ...]  # s-1hw
            save_movie(data=in_, save_path=os.path.join(test_demo_save_path, 'truth.avi'))
            save_movie(data=out_, save_path=os.path.join(test_demo_save_path, 'pred.avi'))
            save_movie(data=in_out_, save_path=os.path.join(test_demo_save_path, 'truth_pred.avi'))
            save_image(data=in_, save_path=os.path.join(test_demo_save_path, 'truth_img'))
            save_image(data=out_, save_path=os.path.join(test_demo_save_path, 'pred_img'))
            save_image(data=in_out_, save_path=os.path.join(test_demo_save_path, 'truth_pred_img'))
        print('%d save movies and images done!' % (i + 1))
