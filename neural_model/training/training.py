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

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num

    def forward(self, preds, labels):

        crossEntropy = nn.CrossEntropyLoss()

        loss0 = crossEntropy(preds[0], labels[0])
        loss1 = crossEntropy(preds[1], labels[1])
        loss2 = crossEntropy(preds[2], labels[2])
        loss3 = crossEntropy(preds[3], labels[3])
        
        return (loss0+loss1+loss2+loss3)/4



TEXT = Field(tokenize='spacy', batch_first=True, include_lengths=True)
ITEM_LABEL = LabelField(dtype = torch.long, batch_first=True)
ACTION_LABEL = LabelField(dtype = torch.long, batch_first=True)
COLOR_LABEL = LabelField(dtype = torch.long, batch_first=True)
SIZE_LABEL = LabelField(dtype = torch.long, batch_first=True)

fields = [('command',TEXT), ('item_label', ITEM_LABEL), ('action_label', ACTION_LABEL), ('color_label', COLOR_LABEL), ('size_label', SIZE_LABEL)]


# dataField=[(None,None),("comment_text",TEXT),("toxic",LABEL)]

training_data = TabularDataset(path=join(dirname(dirname(abspath(__file__))), 'data', 'data.csv'),
                               format = 'csv',fields = fields,skip_header = True)

# train_data, valid_data = training_data.split(split_ratio=0.8)
train_data, valid_data = training_data.split(split_ratio=0.7, random_state=None)
# train_data, valid_data = training_data.split(split_ratio=0.7, random_state=random.seed(2021))

# print(vars(training_data.examples[0]))


#initialize glove embeddings
# TEXT.build_vocab(train_data,min_freq=3,vectors = "glove.6B.100d")  
import pickle

# TEXT.build_vocab(train_data, min_freq=3)  
# ITEM_LABEL.build_vocab(train_data)
# ACTION_LABEL.build_vocab(train_data)
# COLOR_LABEL.build_vocab(train_data)
# SIZE_LABEL.build_vocab(train_data)

if os.path.isfile('built_vocab.pickle'):
    with open('built_vocab.pickle', 'rb') as handle:
        TEXT, ITEM_LABEL, ACTION_LABEL, COLOR_LABEL, SIZE_LABEL = pickle.load(handle)
else:
    with open('built_vocab.pickle', 'wb') as handle:
        TEXT.build_vocab(train_data)  
        ITEM_LABEL.build_vocab(train_data)
        ACTION_LABEL.build_vocab(train_data)
        COLOR_LABEL.build_vocab(train_data)
        SIZE_LABEL.build_vocab(train_data)
        pickle.dump([TEXT, ITEM_LABEL, ACTION_LABEL, COLOR_LABEL, SIZE_LABEL], handle, protocol=pickle.HIGHEST_PROTOCOL)






# l = len(ACTION_LABEL.vocab.itos)
output_dims = {
    'item': len(ITEM_LABEL.vocab),
    'action': len(ACTION_LABEL.vocab),
    'color': len(COLOR_LABEL.vocab),
    'size': len(SIZE_LABEL.vocab)
}


BATCH_SIZE = 256

#Load an iterator
train_iterator, valid_iterator = BucketIterator.splits(
    (train_data, valid_data), 
    batch_size = BATCH_SIZE,
    sort_key = lambda x: len(x.command),
    sort_within_batch=True,
    device = device)

# import torch.nn as nn

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
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

import torch.optim as optim

#define optimizer and loss
optimizer = optim.Adam(model.parameters())
# optimizer = torch.optim.SGD(model.parameters(), lr=5)
criterion = MultiTaskLossWrapper(task_num=4)

#define metric
def _accuracy(preds, labels):
    #round predictions to the closest integer
    rounded_preds =  [torch.argmax(pred, dim=1) for pred in preds]
    correct2 = [(rounded_pred == label).float() for rounded_pred, label in zip(rounded_preds, labels)] 
    acc2 = torch.tensor([c.sum() for c in correct2]) 
    return acc2/rounded_preds[0].numel()
    
#push to cuda if available
model = model.to(device)
criterion = criterion.to(device)

def train(model, iterator, optimizer, criterion):
    
    #initialize every epoch 
    epoch_loss = 0
    epoch_acc = 0
    epoch_task_acc = torch.tensor([0,0,0,0], dtype=torch.float32)
    
    #set the model in training phase
    model.train()  
    
    for batch in iterator:
        
        #resets the gradients after every batch
        optimizer.zero_grad()   
        
        #retrieve text and no. of words
        text, text_lengths = batch.command
        
        #convert to 1D tensor
        predictions = model(text, text_lengths)
        # actions, item, color, size = predictions
        # predictions = torch.argmax(predictions, dim=1)
        #compute the loss
        loss = criterion(predictions, [batch.item_label, batch.action_label, batch.color_label, batch.size_label])        
        
        #compute the binary accuracy
        acc = _accuracy(predictions, [batch.item_label, batch.action_label, batch.color_label, batch.size_label])   
        # print('==================== Acuuracy ====================')
        # print(f'Action: {acc[0]:.2f}, Size: {acc[1]:.2f}, Color: {acc[2]:.2f}, Item: {acc[2]:.2f}')
        #backpropage the loss and compute the gradients
        loss.backward()       
        
        #update the weights
        optimizer.step()      
        
        #loss and accuracy
        epoch_task_acc[0] += acc[0]
        epoch_task_acc[1] += acc[1]
        epoch_task_acc[2] += acc[2]
        epoch_task_acc[3] += acc[3]
        
        acc = (acc[0] + acc[1] + acc[2] + acc[3])/4
        epoch_loss += loss.item()  
        epoch_acc += acc.item()    
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_task_acc / len(iterator)

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
    return rounded_pred

def evaluate(model, iterator, criterion):
    
    #initialize every epoch
    epoch_loss = 0
    epoch_acc = 0
    epoch_task_acc = torch.tensor([0,0,0,0], dtype=torch.float32)

    #deactivating dropout layers
    model.eval()
    
    #deactivates autograd
    with torch.no_grad():
    
        for batch in iterator:
            
            #retrieve text and no. of words
            text, text_lengths = batch.command
            
            #convert to 1D tensor
            predictions = model(text, text_lengths)
            # actions, item, color, size = predictions
            # predictions = torch.argmax(predictions, dim=1)
            #compute the loss
            loss = criterion(predictions, [batch.item_label, batch.action_label, batch.color_label, batch.size_label])        
            
            #compute the binary accuracy
            acc = _accuracy(predictions, [batch.item_label, batch.action_label, batch.color_label, batch.size_label])   
            epoch_task_acc[0] += acc[0]
            epoch_task_acc[1] += acc[1]
            epoch_task_acc[2] += acc[2]
            epoch_task_acc[3] += acc[3]
            acc = (acc[0] + acc[1] + acc[2] + acc[3])/4

            #keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_task_acc / len(iterator)

def run_train():
    N_EPOCHS = 20
    train_info = np.zeros((4, N_EPOCHS))
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        
        #train the model
        train_loss, train_acc, train_task_acc = train(model, train_iterator, optimizer, criterion)
        writer.add_scalar("Train Loss", train_loss, epoch)
        writer.add_scalar("Train Accuracy", train_acc, epoch)

        # writer.add_scalar("Train Loss", train_loss, epoch)


        
        #evaluate the model
        valid_loss, valid_acc, valid_task_acc = evaluate(model, valid_iterator, criterion)
        train_info[0, epoch] = train_loss
        train_info[1, epoch] = valid_loss
        train_info[2, epoch] = train_acc
        train_info[3, epoch] = valid_acc
        writer.add_scalar("Valid Loss", valid_loss, epoch)
        writer.add_scalar("Valid Accuracy", valid_acc, epoch)

        writer.add_scalars("Loss", {'train loss': train_loss, 'validation loss': valid_loss}, epoch)
        writer.add_scalars("Accuracy", {'train accuracy': train_acc, 'validation accuracy': valid_acc}, epoch)



        #save the best model
        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), join(dirname(__file__), 'saved_weights2.pt'))
        
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Item: {train_task_acc[0]:.4f}, Action: {train_task_acc[1]:.4f}, Color: {train_task_acc[2]:.4f}, Size: {train_task_acc[2]:.4f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% | Item: {valid_task_acc[0]:.4f}, Action: {valid_task_acc[1]:.4f}, Color: {valid_task_acc[2]:.4f}, Size: {valid_task_acc[2]:.4f}')
    # np.save(os.path.join(os.getcwd(), 'analysis', f'lstm-{num_layers}-{"bi" if bidirectional is True else "non_bi"}.npy'), train_info)
    
# run_train()
# # print(count_parameters(model))
# writer.flush()
# writer.close()

path='saved_weights2.pt'
model.load_state_dict(torch.load(join(dirname(__file__), path)))
model.eval()

# samples = '''can you bring the small white bread knife from the fridge 
# can you grab the white measuring cups 
# put that small white thing in the basket
# hold that baking sheet pan close
# could you please bring me the red mushrooms  
# could you please hold the green potatoes 
# could you please grab the large white ladle 
# can you grab the small yellow large pot 
# could you put the large white whisk on the table
# could you hand me the large red blender '''.split('\n')

# for sample in samples:
#     s=predict(model, sample)
#     print(sample)
#     print(ITEM_LABEL.vocab.itos[s[0]])
#     print(ACTION_LABEL.vocab.itos[s[1]])
#     print(COLOR_LABEL.vocab.itos[s[2]])
#     print(SIZE_LABEL.vocab.itos[s[3]])
#     print('========================')