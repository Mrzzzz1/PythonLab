import torch
from torch import nn
embedding_dim = 256
class MyModule(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(MyModule, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1
        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers,
                                bidirectional=bidirectional,dropout=0.7)
        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions,
                            batch_size, self.hidden_size)
        return hidden
    def forward(self, input):
        #input = input.t()
        batch_size = input.size(1)

        hidden = self._init_hidden(batch_size)
        embedding = self.embedding(input)

        output, hidden = self.gru(embedding, hidden)
        #print(hidden.size())
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]
        #print(hidden_cat.size())
        fc_output = self.fc(hidden_cat)

        return fc_output