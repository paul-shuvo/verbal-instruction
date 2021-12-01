# from data import COMM_DATA
from os.path import join 
from os import getcwd
from os.path import dirname, abspath
import torch
from torch import nn
# from nltk.tokenize import sent_tokenize, word_tokenize
import random
from torchtext.legacy.data import TabularDataset, Field, LabelField, BucketIterator
import spacy
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import os

import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

SEED = 2021

torch.manual_seed(SEED)



TEXT = Field(tokenize='spacy', batch_first=True, include_lengths=True)
ITEM_LABEL = LabelField(dtype = torch.long, batch_first=True)
ACTION_LABEL = LabelField(dtype = torch.long, batch_first=True)
COLOR_LABEL = LabelField(dtype = torch.long, batch_first=True)
SIZE_LABEL = LabelField(dtype = torch.long, batch_first=True)

fields = [('command',TEXT), ('item_label', ITEM_LABEL), ('action_label', ACTION_LABEL), ('color_label', COLOR_LABEL), ('size_label', SIZE_LABEL)]



import pickle

# TEXT.build_vocab(train_data, min_freq=3)  
# ITEM_LABEL.build_vocab(train_data)
# ACTION_LABEL.build_vocab(train_data)
# COLOR_LABEL.build_vocab(train_data)
# SIZE_LABEL.build_vocab(train_data)


with open(join(dirname(__file__),'built_vocab.pickle'), 'rb') as handle:
    TEXT, ITEM_LABEL, ACTION_LABEL, COLOR_LABEL, SIZE_LABEL = pickle.load(handle)

# l = len(ACTION_LABEL.vocab.itos)
output_dims = {
    'item': len(ITEM_LABEL.vocab),
    'action': len(ACTION_LABEL.vocab),
    'color': len(COLOR_LABEL.vocab),
    'size': len(SIZE_LABEL.vocab)
}

class Classifier(nn.Module):
    
    #define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dims, n_layers, 
                 bidirectional):
        
        #Constructor
        super().__init__()          
        
        #embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        #lstm layer
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional,
                           batch_first=True)
        
        #dense layer
        d = 2 if bidirectional else 1
        self.bidirectional = bidirectional
        
        self.fc_item = nn.Linear(hidden_dim * d, output_dims['item'])
        self.fc_action = nn.Linear(hidden_dim * d, output_dims['action'])
        self.fc_color = nn.Linear(hidden_dim * d, output_dims['color'])
        self.fc_size = nn.Linear(hidden_dim * d, output_dims['size'])
        
        #activation function
        self.act = nn.Softmax(dim=1)
        
    def forward(self, text, text_lengths):
        
        #text = [batch size,sent_length]
        embedded = self.embedding(text)
        #embedded = [batch size, sent_len, emb dim]
      
        #packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths=text_lengths.cpu(), batch_first=True)
        # s = self.lstm(packed_embedded)
        if isinstance(self.lstm, torch.nn.modules.rnn.LSTM):
            packed_output, (hidden, cell) = self.lstm(packed_embedded)
        else:
            packed_output, hidden = self.lstm(packed_embedded)
        #hidden = [batch size, num layers * num directions, hid dim]
        #cell = [batch size, num layers * num directions, hid dim]
        
        #concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1) if self.bidirectional else hidden[-1, :, :]
        #hidden = [batch size, hid dim * num directions]
        item_outputs = self.fc_item(hidden)
        action_outputs=self.fc_action(hidden)
        color_outputs = self.fc_color(hidden)
        size_outputs = self.fc_size(hidden)


        #Final activation function
        # outputs=self.act(dense_outputs)
        
        return [self.act(item_outputs), self.act(action_outputs), self.act(color_outputs), self.act(size_outputs)]
    
#define hyperparameters
size_of_vocab = len(TEXT.vocab)
embedding_dim = 100# 100
num_hidden_nodes = 32 # 32
num_output_nodes = output_dims
num_layers = 1 # 2
bidirectional = True
# dropout = 0

#instantiate the model
model = Classifier(size_of_vocab, embedding_dim, num_hidden_nodes,num_output_nodes, num_layers, 
                   bidirectional = bidirectional)

#architecture
# print(model)

#No. of trianable parameters



    
#push to cuda if available
model = model.to(device)

nlp = spacy.load("en_core_web_sm")

def predict(model, sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]  #tokenize the sentence 
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]          #convert to integer sequence
    length = [len(indexed)]                                    #compute no. of words
    tensor = torch.LongTensor(indexed).to(device)              #convert to tensor
    tensor = tensor.unsqueeze(1).T                             #reshape in form of batch,no. of words
    length_tensor = torch.LongTensor(length)                   #convert to tensor
    preds = model(tensor, length_tensor)  
    rounded_pred =  [torch.argmax(pred, dim=1) for pred in preds]

    # prediction = torch.argmax(prediction, dim=1)               #prediction 
    task_params = {}
    task_params['item'] = ITEM_LABEL.vocab.itos[rounded_pred[0]]
    task_params['action'] = ACTION_LABEL.vocab.itos[rounded_pred[1]]
    task_params['color'] = COLOR_LABEL.vocab.itos[rounded_pred[2]]
    task_params['size'] = SIZE_LABEL.vocab.itos[rounded_pred[3]]

    return task_params


path='saved_weights.pt'
model.load_state_dict(torch.load(join(dirname(__file__), path)))
model.eval()

# samples = '''can you bring me the small black oven mitts
# go left'''.split('\n')

# for sample in samples:
#     s=predict(model, sample)
#     print(sample)
#     print(s)
#     # print(ITEM_LABEL.vocab.itos[s[0]])
#     # print(ACTION_LABEL.vocab.itos[s[1]])
#     # print(COLOR_LABEL.vocab.itos[s[2]])
#     # print(SIZE_LABEL.vocab.itos[s[3]])
#     print('========================')