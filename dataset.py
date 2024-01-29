
# -*- coding: utf-8 -*-
"""
Created on 31/3/2019
@author: RuihongQiu
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
import torch
from torch_geometric.data import InMemoryDataset, Dataset ,Data
from tqdm import tqdm

class MultiSessionsGraph(InMemoryDataset):
    """Every session is a graph."""
    def __init__(self, root, phrase, transform=None, pre_transform=None):
        """
        Args:
            root: 'sample', 'yoochoose1_4', 'yoochoose1_64' or 'diginetica'
            phrase: 'train' or 'test'
        """
        assert phrase in ['train', 'test']
        self.phrase = phrase
        super(MultiSessionsGraph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
     
    @property
    def raw_file_names(self):
        return [self.phrase + '.txt']
    
    @property
    def processed_file_names(self):
        return [self.phrase + '.pt']
    
    def download(self):
        pass
    
    def process(self):
        data = pickle.load(open(self.raw_dir + '\\' + self.raw_file_names[0], 'rb'))
        data_list = []
        
        for sequences, y in zip(data[0], data[1]):
            i = 0
            nodes = {}    # dict{15: 0, 16: 1, 18: 2, ...}
            senders = []
            x = []
            for node in sequences:
                if node not in nodes:
                    nodes[node] = i
                    x.append([node])
                    i += 1
                senders.append(nodes[node])
            receivers = senders[:]
            del senders[-1]    # the last item is a receiver
            del receivers[0]    # the first item is a sender
            edge_index = torch.tensor([senders, receivers], dtype=torch.long)
            x = torch.tensor(x, dtype=torch.long)
            y = torch.tensor([y], dtype=torch.long)
            data_list.append(Data(x=x, edge_index=edge_index, y=y))
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# fix from here        
datatset_size = 1000

clicks_df = pd.read_csv('data/yoochoose/yoochoose-clicks.dat', header=None)
clicks_df.columns = ['session_id', 'timestamp', 'item_id']
print(clicks_df.head(5))

buy_df = pd.read_csv('data/yoochoose/yoochoose-buys.dat', header=None)
buy_df.columns = ['session_id', 'timestamp', 'item_id', 'price', 'quantity']
print(buy_df.head(5))



#randomly sample a couple of them
sampled_session_id = np.random.choice(clicks_df.session_id.unique(), datatset_size, replace=False)
clicks_df = clicks_df.loc[clicks_df.session_id.isin(sampled_session_id)]
print(clicks_df.nunique())

item_encoder = LabelEncoder()
clicks_df['item_id'] = item_encoder.fit_transform(clicks_df['item_id'].astype(int))
print(clicks_df.head())

clicks_df['label'] = clicks_df.session_id.isin(buy_df.session_id)
print(clicks_df.head())

print(clicks_df.item_id.max() + 1)


class YooChooseBinaryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(YooChooseBinaryDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load('./data/processed.dataset')

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data/yoochoose_click_binary_1M_sess.dataset']

    def download(self):
        pass

    def process(self):

        data_list = []

        # process by session_id
        grouped = clicks_df.groupby('session_id')
        for session_id, group in tqdm(grouped):
            sess_item_id = LabelEncoder().fit_transform(group.item_id)
            group = group.reset_index(drop=True)
            group['sess_item_id'] = sess_item_id
            node_features = group.loc[group.session_id == session_id, ['sess_item_id', 'item_id']].sort_values('sess_item_id').item_id.drop_duplicates().values

            node_features = torch.LongTensor(node_features).unsqueeze(1)
            target_nodes = group.sess_item_id.values[1:]
            source_nodes = group.sess_item_id.values[:-1]

            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            x = node_features

            y = torch.FloatTensor([group.label.values[0]])

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), './data/processed.dataset')


# The class inherits the base class Dataset from pytorch
class LoadData(Dataset):  # for training/testing
    def __init__(self, data_path):
        super(LoadData, self).__init__()
        self.data = torch.load(data_path)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':

    dataset = YooChooseBinaryDataset('./')
    one_tenth_length = int(len(dataset) * 0.1)
    dataset = dataset.shuffle()
    train_dataset = dataset[:one_tenth_length * 8]
    val_dataset = dataset[one_tenth_length * 8:one_tenth_length * 9]
    test_dataset = dataset[one_tenth_length * 9:]
    print(len(train_dataset), len(val_dataset), len(test_dataset))