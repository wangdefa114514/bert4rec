import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import defaultdict
import numpy as np
import os

import torch
from torch.utils.data import Dataset
import pandas as pd
from collections import defaultdict
import os
import pickle

import torch
from torch.utils.data import Dataset
import pandas as pd
from collections import defaultdict
import os
import pickle
import numpy as np

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
from collections import defaultdict
import random
from tqdm import tqdm

class MovieLensDataset(Dataset):
    def __init__(self, data_dir, max_len=50, split='train', test_ratio=0.2, random_seed=42, mask_prob=0.15):
        self.data_dir = data_dir
        self.max_len = max_len
        self.split = split
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.mask_prob = mask_prob
        self.processed_file = os.path.join(data_dir, f'processed_data_{max_len}_{test_ratio}_{random_seed}.pkl')

        if os.path.exists(self.processed_file):
            self.load_processed_data()
        else:
            self.process_and_save_data()

        self.rng = random.Random(self.random_seed)

    def process_and_save_data(self):
        # 读取电影数据
        movies_file = os.path.join(self.data_dir, 'movies.dat')
        self.movies_df = pd.read_csv(movies_file, sep='::', names=['movie_id', 'title', 'genres'], engine='python', encoding='latin1')
        self.movie_id_to_idx = {id: idx + 1 for idx, id in enumerate(self.movies_df['movie_id'].unique())}  # 从1开始编号
        self.num_items = len(self.movie_id_to_idx) + 1  # +1 for padding
        self.mask_token = self.num_items + 1  # 最后一个索引作为mask token
        
        # 读取评分数据
        ratings_file = os.path.join(self.data_dir, 'ratings.dat')
        ratings_df = pd.read_csv(ratings_file, sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python')
        ratings_df = ratings_df.sort_values(['user_id', 'timestamp'])
        
        # 构建用户观影序列
        user_sequences = ratings_df.groupby('user_id').apply(
            lambda x: x.sort_values('timestamp')['movie_id'].map(self.movie_id_to_idx).tolist()
        ).to_dict()

        # 计算物品流行度（用于负采样）
        self.item_popularity = ratings_df['movie_id'].map(self.movie_id_to_idx).value_counts().to_dict()

       # 确保所有电影都有一个流行度值（即使是0）
        for movie_id in self.movie_id_to_idx.values():
            if movie_id not in self.item_popularity:
                self.item_popularity[movie_id] = 0
        
        # 划分训练集、验证集和测试集
        train_sequences = []
        val_sequences = []
        val_targets = []
        test_sequences = []
        test_targets = []
        negative_samples = {}

        np.random.seed(self.random_seed)

        for user_id, sequence in tqdm(user_sequences.items()):
            if len(sequence) > 2:  # 至少需要3个item才能划分
                if len(sequence) > self.max_len:
                    sequence = sequence[-self.max_len:]  # 只保留最后max_len个项目
                
                train_seq = sequence[:-2]
                val_seq = sequence[:-1]
                test_seq = sequence

                train_sequences.append((user_id, train_seq))
                val_sequences.append(val_seq)
                val_targets.append(val_seq[-1])
                test_sequences.append(test_seq)
                test_targets.append(test_seq[-1])

                # 为每个用户生成100个负样本
                neg_samples = self.sample_negative_items(sequence[-1], 100)
                negative_samples[user_id] = neg_samples
        
        # 保存处理后的数据
        with open(self.processed_file, 'wb') as f:
            pickle.dump((train_sequences, val_sequences, val_targets, test_sequences, test_targets, negative_samples, self.movie_id_to_idx, self.item_popularity, self.num_items, self.mask_token), f)
        self.train_sequences = train_sequences
        self.val_sequences = val_sequences
        self.val_targets = val_targets
        self.test_sequences = test_sequences
        self.test_targets = test_targets
        self.negative_samples = negative_samples
        self.set_split(self.split)

    def load_processed_data(self):
        with open(self.processed_file, 'rb') as f:
            train_sequences, val_sequences, val_targets, test_sequences, test_targets, negative_samples, self.movie_id_to_idx, self.item_popularity, self.num_items, self.mask_token = pickle.load(f)
        
        
        self.train_sequences = train_sequences
        self.val_sequences = val_sequences
        self.val_targets = val_targets
        self.test_sequences = test_sequences
        self.test_targets = test_targets
        self.negative_samples = negative_samples
        self.set_split(self.split)

    def set_split(self, split):
        self.split = split
        if split == 'train':
            self.sequences = self.train_sequences
        elif split == 'val':
            self.sequences = self.val_sequences
            self.targets = self.val_targets
        elif split == 'test':
            self.sequences = self.test_sequences
            self.targets = self.test_targets

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.split == 'train':
            user, seq = self.sequences[idx]
            tokens = []
            labels = []
            for s in seq:
                prob = self.rng.random()
                if prob < self.mask_prob:
                    prob /= self.mask_prob

                    if prob < 0.8:
                        tokens.append(self.mask_token)
                    elif prob < 0.9:
                        tokens.append(self.rng.randint(1, self.num_items))
                    else:
                        tokens.append(s)

                    labels.append(s)
                else:
                    tokens.append(s)
                    labels.append(0)

            tokens = tokens[-self.max_len:]
            labels = labels[-self.max_len:]

            mask_len = self.max_len - len(tokens)

            tokens = [0] * mask_len + tokens
            labels = [0] * mask_len + labels

            return torch.LongTensor(tokens), torch.LongTensor(labels)
        else:
            seq = self.sequences[idx]
            answer = [self.targets[idx]]
            negs = self.negative_samples[seq[-1]]
    
            candidates = answer + negs
            labels = [1] * len(answer) + [0] * len(negs)
    
            seq = seq + [self.mask_token]
            seq = seq[-self.max_len:]
            padding_len = self.max_len - len(seq)
            seq = [0] * padding_len + seq
    
            return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)

    def sample_negative_items(self, positive_item, num_samples):
        # 使用 numpy 数组来存储 item_ids 和 item_probs
        item_ids = np.array(list(self.item_popularity.keys()))
        item_probs = np.array(list(self.item_popularity.values()), dtype=float)
        item_probs /= item_probs.sum()

        # 创建一个布尔掩码，排除正样本
        mask = item_ids != positive_item
        item_ids_filtered = item_ids[mask]
        item_probs_filtered = item_probs[mask]
        item_probs_filtered /= item_probs_filtered.sum()  # 重新归一化概率

        # 一次性采样所有负样本，然后去重
        samples = np.random.choice(
            item_ids_filtered, 
            size=num_samples * 2,  # 多采样一些，以防重复
            replace=False, 
            p=item_probs_filtered
        )
        
        # 去重并截断到所需数量
        negative_items = list(dict.fromkeys(samples))[:num_samples]
        
        return negative_items
