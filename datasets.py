import json

import h5py
import torch
from torch.utils.data import Dataset


class ClassifierDataset(Dataset):
    def __init__(self, feature_path, label_path, feature_map_path, split):
        self.feature_path = feature_path
        self.image_features = None
        with open(label_path, 'r') as fp:
            self.dataset = json.load(fp)[split]
        with open(feature_map_path, 'r') as fp:
            self.feature_map = json.load(fp)

    def __getitem__(self, i):
        if not self.image_features:
            self.image_features = h5py.File(self.feature_path,
                                            'r')['image_feature']
        image_id = self.dataset[i]['image_id']
        feature_id = int(self.feature_map[str(image_id)])
        image_feature = self.image_features[feature_id]  #type: ignore
        label = self.dataset[i]['label']
        return {
            'image_id': image_id,
            'inputs': (torch.tensor(image_feature), ),
            'label': (torch.tensor(label, dtype=torch.float32), ),
        }

    def __len__(self):
        return len(self.dataset)


class LSTMDataset(Dataset):
    def __init__(self, feature_path, label_path, feature_map_path, split):
        self.feature_path = feature_path
        self.image_features = None
        with open(label_path, 'r') as fp:
            self.dataset = json.load(fp)[split]
        with open(feature_map_path, 'r') as fp:
            self.feature_map = json.load(fp)

    def __getitem__(self, i):
        if not self.image_features:
            self.image_features = h5py.File(self.feature_path,
                                            'r')['image_feature']
        image_id = self.dataset[i]['image_id']
        feature_id = int(self.feature_map[str(image_id)])
        image_feature = torch.tensor(
            self.image_features[feature_id])  #type: ignore
        seq = torch.tensor(self.dataset[i]['seq'])
        seq_length = torch.tensor(self.dataset[i]['seq_length'])
        return {
            'image_id': image_id,
            'inputs': (image_feature, seq, seq_length),
            'label': (seq, seq_length)
        }

    def __len__(self):
        return len(self.dataset)
