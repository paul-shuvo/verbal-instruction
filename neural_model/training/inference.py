import torch
from torch import nn
from torchtext.legacy.data import TabularDataset, Field, LabelField, BucketIterator
from os.path import join, dirname, abspath
# from torch.le
# model = 
SEED = 2021
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
TEXT = Field(tokenize='spacy', batch_first=True, include_lengths=True)
LABEL = LabelField(dtype = torch.long, batch_first=True)

class Classifier(nn.Module):
    #define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dims, n_layers, 
                 bidirectional, dropout):
        
        #Constructor
        super().__init__()          
        
        #embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        #lstm layer
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first=True)
        
        #dense layer
        d = 2 if bidirectional else 1
        self.bidirectional = bidirectional
        
        self.fc_action = nn.Linear(hidden_dim * d, output_dims['action'])
        self.fc_item = nn.Linear(hidden_dim * d, output_dims['item'])
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
        action_outputs=self.fc_action(hidden)
        size_outputs = self.fc_size(hidden)
        color_outputs = self.fc_color(hidden)
        item_outputs = self.fc_item(hidden)


        #Final activation function
        # outputs=self.act(dense_outputs)
        
        return [self.act(action_outputs), self.act(size_outputs), self.act(color_outputs), self.act(item_outputs)]
    

actions = ['give',
        #    'give',
        #    'hand me',
           'hand',
        #    'bring me',
           'bring',
           'put',
           'grab',
           'hold',
           'measure']
# TEXT= Field(sequential=True,lower=True,tokenize=Tokenizer,eos_token='EOS',stop_words=nlp.Defaults.stop_words,include_lengths=True)
 
# LABEL= Field(dtype=torch.float,sequential=False,use_vocab=False,pad_token=None,unk_token=None)

fields = [('label', LABEL), ('text',TEXT)]

training_data = TabularDataset(path=join(dirname(dirname(abspath(__file__))), 'data', 'data.csv'),format = 'csv',fields = fields,skip_header = True)

train_data, valid_data = training_data.split(split_ratio=0.7)
fields = [('label', LABEL), ('text',TEXT)]



TEXT.build_vocab(train_data, min_freq=3)  
LABEL.build_vocab(train_data)

#define hyperparameters
size_of_vocab = len(TEXT.vocab)
embedding_dim = 30 # 100
num_hidden_nodes = 32 # 32
num_output_nodes = len(LABEL.vocab)
num_layers = 2 # 2
bidirection = True
dropout = 0.4

#instantiate the model
model = Classifier(size_of_vocab, embedding_dim, num_hidden_nodes,num_output_nodes, num_layers, 
                   bidirectional = True, dropout = dropout)
model = model.to(device)

import spacy
nlp = spacy.load('en')
#load weights
path_ = join(dirname(dirname(__file__)), 'neural_model', 'weights', 'saved_weights.pt')
model.load_state_dict(torch.load(path_))
model.eval()

#inference 

def predict(model, sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]  #tokenize the sentence 
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]          #convert to integer sequence
    length = [len(indexed)]                                    #compute no. of words
    tensor = torch.LongTensor(indexed).to(device)              #convert to tensor
    tensor = tensor.unsqueeze(1).T                             #reshape in form of batch,no. of words
    length_tensor = torch.LongTensor(length)                   #convert to tensor
    prediction = model(tensor, length_tensor)  
    prediction = torch.argmax(prediction, dim=1)#prediction 
    return prediction.item()

#make predictions
samples = '''can you bring the small white bread knife from the fridge 
can you grab the white measuring cups 
put that small white thing in the basket
hold that baking sheet pan close
could you please bring me the red mushrooms  
could you please hold the green potatoes 
could you please grab the large white ladle 
can you grab the small yellow large pot 
could you put the large white whisk on the table
could you hand me the large red blender '''.split('\n')

for s in samples:
    print(f'Sample: {s}, Prediction: {LABEL.vocab.itos[predict(model, s)]}')

#insincere question
# print(actions[predict(model, "can you put the small red cast iron skillet in the bowl")])