import torch
from torch import nn
from torch.utils.data import Dataset  


# function to predict accuracy
def accuracy(pred,label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()


class SentimentDataset(Dataset):

    def __init__(self, reviews, labels):
        self.reviews = reviews
        self.labels = labels

    def __len__(self):
        return self.reviews.shape[0]
    
    def __getitem__(self, idx):
        return self.reviews[idx, :], self.labels[idx]
    


class SentimentLSTM(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(SentimentLSTM, self).__init__()
        
        # define the params
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        

        # defining the basic building blocks
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embedding_dim)  
        self.lstm = nn.LSTM(input_size = embedding_dim, 
                            hidden_size = hidden_dim, 
                            dropout = drop_prob, 
                            num_layers = n_layers,  
                            batch_first=True) 
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(in_features = hidden_dim, out_features = output_size)  
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        # size of x is [BATCH_SIZE, SEQUENCE_LENGTH]
        batch_size = x.size(0)

        out = self.embedding(x)  # size of out is [BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_DIM]
        out, hidden = self.lstm(out, hidden) # size of out is [BATCH_SIZE, SEQUENCE_LENGTH, LSTM_HIDDEN_DIM]

        # hidden will have 2 tenors (hidden state & cell state of size [NUM_LAYERS, BATCH, LSTM_HIDDEN_DIM])

        # stack up lstm outputs
        # Get the output of the last time step
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = out[:,-1,:]
        # dropout and fully-connected layer
        out = self.dropout(out)
        out = self.fc(out)
        out = torch.sigmoid(out)
        
        # return last sigmoid output and hidden state
        return out, hidden


    def init_hidden(self, batch_size, device):
        ''' Initializes hidden state '''
        # Create two new tensors (hidden state, and cell state) with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
    
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                    weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
    
        return hidden
