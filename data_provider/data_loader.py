
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None, scale=True):
        """
        root_path: 数据文件路径 (例如 './data/PEMS.npz')
        flag: 'train', 'val', 'test'
        size: [seq_len, label_len, pred_len]
        scale: 是否进行标准化
        """
        assert flag in ['train', 'val', 'test']
        self.flag = flag
        self.scale = scale
        self.root_path = root_path
        self.seq_len, self.label_len, self.pred_len = size
        self.type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = self.type_map[flag]

        self.__read_data__()

    def __read_data__(self):
        # 初始化标准化器
        self.scaler = StandardScaler()

        # 读取 npz 文件
        data_file = os.path.join(self.root_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]  # (时间步, 节点, 特征=1)

        # 数据划分
        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)): int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        # 标准化
        if self.scale:
            self.scaler.fit(train_data.flatten().reshape(-1, 1))
            data_scaled = self.scaler.transform(data.flatten().reshape(-1, 1)).reshape(data.shape)
        else:
            data_scaled = data

        # 填补缺失值
        df_scaled = pd.DataFrame(data_scaled)
        df_scaled = df_scaled.fillna(method='ffill', limit=len(df_scaled)).fillna(method='bfill', limit=len(df_scaled)).values

        # 输入与输出都使用标准化后的数据
        self.data_x = df_scaled
        self.data_y = df_scaled

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """将标准化数据反归一化回原始量纲"""
        return self.scaler.inverse_transform(data.flatten().reshape(-1, 1)).reshape(data.shape)
