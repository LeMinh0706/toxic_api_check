import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

'''This code use pre-trained PhoBERT model and CNN model to detect offensive chat messages.'''

class CNN(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx):
        super().__init__()
        
        self.fc_input = nn.Linear(embedding_dim, embedding_dim)
        
        self.conv_0 = nn.Conv1d(in_channels=embedding_dim,
                               out_channels=n_filters,
                               kernel_size=filter_sizes[0])
        
        self.conv_1 = nn.Conv1d(in_channels=embedding_dim,
                               out_channels=n_filters,
                               kernel_size=filter_sizes[1])
        
        self.conv_2 = nn.Conv1d(in_channels=embedding_dim,
                               out_channels=n_filters,
                               kernel_size=filter_sizes[2])
        
        self.conv_3 = nn.Conv1d(in_channels=embedding_dim,
                               out_channels=n_filters,
                               kernel_size=filter_sizes[3])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, encoded):
        embedded = self.fc_input(encoded)
        embedded = embedded.permute(0, 2, 1)
        
        conved_0 = F.relu(self.conv_0(embedded))
        conved_1 = F.relu(self.conv_1(embedded))
        conved_2 = F.relu(self.conv_2(embedded))
        conved_3 = F.relu(self.conv_3(embedded))
        
        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        pooled_3 = F.max_pool1d(conved_3, conved_3.shape[2]).squeeze(2)
        
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2, pooled_3), dim=1))
        
        return self.fc(cat)


