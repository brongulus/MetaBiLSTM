from unicodedata import bidirectional
import torch
import torch.nn as nn


class BiGRU(nn.module):
    def __init__(
        self,
        input_size,
        hidden_dim,
        device,
        num_layers,
        dropout,
    ):
        super().__init__()
        self.input_size = (input_size,)
        self.rnn = nn.GRU(
            input_size=self.input_size,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.device = device
        self.relu = nn.ReLU()

    def forward(self, x):
        inds, lens = x
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            inds,
            lengths=lens,
            batch_first=True,
            enforce_sorted=False,
        )
        output, _ = self.rnn(packed, self.get_initial_state(inds))
        output, lens = torch.nn.utils.rnn.pack_packed_sequence(output, batch_first=True)
        gru_model_output = {
            "logits": output,
            "lens": lens,
        }

        return gru_model_output

    def get_initial_state(self, inp):
        shape = self.rnn.get_expected_hidden_size(inp, None)
        return torch.zeros(shape).to(self.device), torch.zeros(shape).to(self.device)
