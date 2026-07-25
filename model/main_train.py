import os
import logging
import argparse
import torch.multiprocessing


from datetime import datetime

from data_provider.data_factory import data_provider
from model.DGMSTN import make_model

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import shutil
import argparse
import configparser
# from model.MGCN import make_model
from lib.utils import get_adjacency_matrix, compute_val_loss_mgcn, predict_and_save_results,compute_val_loss_former,predict_and_save_results_former
from tensorboardX import SummaryWriter
from lib.metrics import masked_mape_np, masked_mae,masked_mse,masked_rmse

import random
from lib.utils import EarlyStopping, adjust_learning_rate
# ---------------- 全局数据集名称 ----------------
dataset_name_base = "PEMS04"   # <<< 下次只要改这里

model_name = "MambaTest"
# ------------------------------------------------

# pip install  tensorboardX  -i  https://pypi.tuna.tsinghua.edu.cn/simple
# pip install  scikit-learn  -i  https://pypi.tuna.tsinghua.edu.cn/simple

# ---------------- Logger 配置 ----------------
logger = logging.getLogger("train_logger")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

# 日志文件名也和数据集挂钩
log_filename = f"{dataset_name_base}_train.log"
log_dir = f"logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"{dataset_name_base}_train.log")
fh = logging.FileHandler(log_filename, mode="a")
fh.setFormatter(formatter)
logger.addHandler(fh)
# ------------------------------------------------

def _get_data(root_path, flag, seq_len, label_len, pred_len, batch_size):
    data_set, data_loader = data_provider(root_path, flag, seq_len, label_len, pred_len, batch_size)
    return data_set, data_loader


parser = argparse.ArgumentParser(description='[MambaTest] Long Sequences Forecasting')
# parser.add_argument(
#     "--config",
#     default=f'/temp/myMambaTest/model/{model_name}/configurations/{dataset_name_base}.conf',
#     type=str,
#     help="configuration file path"
# )

parser.add_argument(
    "--config",
    default=f'/tmp/myMambaTest/model/MambaTest/configurations/{dataset_name_base}.conf',
    type=str,
    help="configuration file path"
)

args = parser.parse_args()
config = configparser.ConfigParser()
logger.info('Read configuration file: %s' % (args.config))
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None

num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])
dataset_name = data_config['dataset_name']
model_name = training_config['model_name']

ctx = training_config['ctx']
os.environ["CUDA_VISIBLE_DEVICES"] = ctx
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0')
logger.info(f"CUDA: {USE_CUDA}, {DEVICE}")

learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
start_epoch = int(training_config['start_epoch'])
batch_size = int(training_config['batch_size'])
nb_chev_filter = int(training_config['nb_chev_filter'])
nb_time_filter = int(training_config['nb_time_filter'])
in_channels = int(training_config['in_channels'])
K = int(training_config['K'])
loss_function = training_config['loss_function']
metric_method = training_config['metric_method']
missing_value = float(training_config['missing_value'])
time_strides = 1

# 生成带日期的文件夹名
date_str = datetime.now().strftime('%Y%_m%d_%H')  
folder_dir = f'predict{num_for_predict}_{model_name}_{date_str}'
# folder_dir = 'predict%s_Informer' % (num_for_predict)
logger.info(f'folder_dir: {folder_dir}')
params_path = os.path.join('experiments', dataset_name, folder_dir)
logger.info(f'params_path: {params_path}')

train_data, train_loader = _get_data(root_path=graph_signal_matrix_filename, flag='train', seq_len=len_input,
                                     label_len=0, pred_len=num_for_predict, batch_size=batch_size)
vali_data, val_loader = _get_data(root_path=graph_signal_matrix_filename, flag='val', seq_len=len_input,
                                  label_len=0, pred_len=num_for_predict, batch_size=batch_size)
test_data, test_loader = _get_data(root_path=graph_signal_matrix_filename, flag='test', seq_len=len_input,
                                   label_len=0, pred_len=num_for_predict, batch_size=batch_size)

adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)


net = make_model(DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx,
                 num_for_predict, len_input,num_of_vertices,args)
net.to('cuda')
def train_main():
    if (start_epoch == 0) and (not os.path.exists(params_path)):
        os.makedirs(params_path)
        logger.info(f'create params directory {params_path}')
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        logger.info(f'delete the old one and create params directory {params_path}')
    elif (start_epoch > 0) and (os.path.exists(params_path)):
        logger.info(f'train from params directory {params_path}')
    else:
        raise SystemExit('Wrong type of model!')

    early_stopping = EarlyStopping(patience=30)

    logger.info('param list:')
    logger.info(f'CUDA\t {DEVICE}')
    logger.info(f'in_channels\t {in_channels}')
    logger.info(f'nb_chev_filter\t {nb_chev_filter}')
    logger.info(f'nb_time_filter\t {nb_time_filter}')
    logger.info(f'batch_size\t {batch_size}')
    logger.info(f'graph_signal_matrix_filename\t {graph_signal_matrix_filename}')
    logger.info(f'start_epoch\t {start_epoch}')
    logger.info(f'epochs\t {epochs}')

    masked_flag = 0
    criterion = nn.L1Loss().to(DEVICE)
    criterion_masked = masked_mae
    if loss_function == 'masked_mse':
        criterion_masked = masked_mse
        masked_flag = 1
    elif loss_function == 'masked_mae':
        criterion_masked = masked_mae
        masked_flag = 1
    elif loss_function == 'mae':
        criterion = nn.L1Loss().to(DEVICE)
        masked_flag = 0
    elif loss_function == 'rmse':
        criterion = nn.MSELoss().to(DEVICE)
        masked_flag = 0

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    sw = SummaryWriter(logdir=params_path, flush_secs=3)
    logger.info(net)

    # logger.info('Net\'s state_dict:')
    total_param = 0
    for param_tensor in net.state_dict():
        # logger.info(f"{param_tensor}\t {net.state_dict()[param_tensor].size()}")
        total_param += np.prod(net.state_dict()[param_tensor].size())
    logger.info(f'Net\'s total params: {total_param}')

    # logger.info('Optimizer\'s state_dict:')
    # for var_name in optimizer.state_dict():
    #     logger.info(f"{var_name}\t {optimizer.state_dict()[var_name]}")

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf

    time_now = time.time()
    if start_epoch > 0:
        params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)
        net.load_state_dict(torch.load(params_filename))
        logger.info(f'start epoch: {start_epoch}')
        logger.info(f'load weight from: {params_filename}')

    for epoch in range(start_epoch, epochs):
        iter_count = 0
        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

        if masked_flag:
            val_loss = compute_val_loss_mgcn(net, val_loader, criterion_masked, masked_flag, missing_value, sw, epoch, DEVICE)
        else:
            val_loss = compute_val_loss_mgcn(net, val_loader, criterion, masked_flag, missing_value, sw, epoch, DEVICE)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

        net.train()

        for batch_index, (encoder_inputs, labels) in enumerate(train_loader):
            iter_count += 1
            dec_inp = torch.zeros_like(labels[:,-96:, :]).float()
            dec_inp = torch.cat([labels[:, :96, :], dec_inp], dim=1).float()
           
            encoder_inputs = encoder_inputs.float().to(DEVICE)
            labels = labels.float().to(DEVICE)
            optimizer.zero_grad()

            outputs = net(encoder_inputs)
           
            if masked_flag:
                loss = criterion_masked(outputs, labels, missing_value)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            training_loss = loss.item()
            global_step += 1
            sw.add_scalar('training_loss', training_loss, global_step)

            if (batch_index + 1) % 300 == 0:
                logger.info(f"\titers: {batch_index + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                speed = (time.time() - time_now) / iter_count
                logger.info(f"speed: {speed:.6f} s/batch")
                allocated_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
                cached_memory = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)
                total = allocated_memory + cached_memory
                logger.info(f'allocated_memory: {allocated_memory:.4f} GB')
                logger.info(f'cached_memory: {cached_memory:.4f} GB')
                logger.info(f'total: {total:.4f} GB')
                iter_count = 0
                time_now = time.time()

        early_stopping(val_loss, net, params_filename)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break
        adjust_learning_rate(optimizer, epoch + 1, learning_rate)
    logger.info(f'best epoch: {best_epoch}')
    logger.info(f'val loss: {best_val_loss}')
    predict_main(best_epoch, test_loader, test_data, num_for_predict, metric_method, 'test')


def predict_main(global_step, data_loader, test_data, pred_len, metric_method, type):
    params_filename = os.path.join(params_path, 'epoch_%s.params' % global_step)
    logger.info(f'load weight from: {params_filename}')
    net.load_state_dict(torch.load(params_filename))
    predict_and_save_results(logger,net, data_loader, test_data, pred_len, global_step, metric_method, params_path, type)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    fix_seed = 2025
    set_seed(fix_seed)
    train_main()