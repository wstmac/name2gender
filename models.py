import torch
import torch.nn as nn
from torch.autograd import Variable
import configuration


args = configuration.parser_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class RNN(nn.Module):
#     """Recurrent Neural Network
#     Args:
#         input_size: (int) size of data
#         hidden_size: (int) number of hidden units
#         output_size: (int) size of output
#     """
#     def __init__(self, input_size, hidden_size, output_size):
#         super(RNN, self).__init__()
        
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
        
#         self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
#         self.i2o = nn.Linear(input_size + hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)
    
#     def forward(self, names, name_lengths):

#         batch_size = names.size(0)

#         # Sort input data by decreasing lengths
#         name_lengths, sort_ind = name_lengths.squeeze(1).sort(dim=0, descending=True)
#         names = names[sort_ind]

#         # Initialize hidden state
#         hidden = self.init_hidden(batch_size)  # (batch_size, hidden_size)

#         # Create tensors to hold word predicion scores and alphas
#         predictions = torch.zeros(batch_size, self.output_size).to(device)

#         encoder_lengths = name_lengths.tolist()

#         for t in range(max(encoder_lengths)):
#             batch_size_t = sum([l > t for l in encoder_lengths])
#             combined = torch.cat((names[:batch_size_t], hidden[:batch_size_t]), 1)
#             hidden = self.i2h(combined)
#             output = self.i2o(combined)
#             output = self.softmax(output)
#             predictions[:batch_size_t,:] =  output

#         return predictions, sort_ind


#     def init_hidden(self, batch_size):
#         h = torch.zeros(batch_size, self.hidden_size).to(device)
#         return h


class LSTM(nn.Module):
    """LSTM network
    Args:
        input_size: (int) size of data
        hidden_size: (int) number of hidden units
        output_size: (int) size of output
        self.dropout: (float) dropout rate, default 0.5
    """
    def __init__(self, input_size, hidden_size, output_size, dropout=0.5):
        super(LSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout

        self.encoder = nn.LSTMCell(input_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(p=self.dropout)
        self.fc = nn.Linear(hidden_size, output_size)

        if args.disable_he_initialization:
            self.init_weights() # initialize some layers with the He initialization

        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, names, name_lengths):

        batch_size = names.size(0)

        # Sort input data by decreasing lengths
        name_lengths, sort_ind = name_lengths.sort(dim=0, descending=True)
        names = names[sort_ind]

        # Initialize hidden state
        h, c = self.init_hidden(batch_size)  # (batch_size, hidden_size)

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, self.output_size).to(device)

        encoder_lengths = name_lengths.tolist()

        # for each time step t, we encode t's character
        # if the name lenght is smaller than t,
        # then it will not be envolved in the lstm 
        for t in range(max(encoder_lengths)):
            batch_size_t = sum([l > t for l in encoder_lengths])
            h, c = self.encoder(names[:batch_size_t], (h[:batch_size_t], c[:batch_size_t]))
            output = self.fc(h)
            output = self.softmax(output)
            predictions[:batch_size_t,:] =  output

        return predictions, sort_ind


    def init_hidden(self, batch_size):
        h = torch.empty(batch_size, self.hidden_size).to(device)
        c = torch.empty(batch_size, self.hidden_size).to(device)

        if not args.disable_he_initialization: # init with He initialization
            nn.init.kaiming_uniform_(h)
            nn.init.kaiming_uniform_(c)

        return h, c

    def init_weights(self):
        """
        Initializes some parameters with He initialization, for easier convergence.
        """
        nn.init.kaiming_normal_(self.fc.weight)