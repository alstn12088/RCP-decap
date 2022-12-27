import torch
import numpy as np

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, input_graph):
        super(GraphDataset, self).__init__()
        self.graph_list = input_graph

    def __getitem__(self, idx):
        graph = self.graph_list[idx]
 
        return graph

    def __len__(self):
        return len(self.graph_list)
    @staticmethod
    def collate(data_list):
        return torch.tensor(data_list)

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, input_sequence):
        super(SequenceDataset, self).__init__()
        self.sequence_list = input_sequence

    def __getitem__(self, idx):
        sequence = self.sequence_list[idx]
 
        return sequence

    def __len__(self):
        return len(self.sequence_list)
    @staticmethod
    def collate(data_list):
        return torch.tensor(data_list).long()

class ScoreDataset(torch.utils.data.Dataset):
    def __init__(self, input_score,mean,std):
        super(ScoreDataset, self).__init__()
        self.score_list = input_score
        self.scores = torch.FloatTensor(input_score)
        self.mean = mean  
        self.std = std 

    def __getitem__(self, idx):
   
        return self.score_list[idx] 

    def get_tsrs(self):
        return self.scores


    def __len__(self):
        return len(self.score_list)
    @staticmethod
    def collate(data_list):
        return torch.tensor(data_list)
class ZipDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        return [dataset[idx] for dataset in self.datasets]
    @staticmethod
    def collate(data_list):
        return [dataset.collate(data_list) for dataset, data_list in zip(self.datasets, zip(*data_list))]
