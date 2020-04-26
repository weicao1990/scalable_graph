import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

import numpy as np


class Encoder(nn.Module):
    def __init__(self, num_features, num_timesteps_input, hidden_size, overlap_size):
        super(Encoder, self).__init__()

        self.num_features = num_features
        self.overlap_size = overlap_size

        rnn_input_size = num_features * overlap_size

        self.rnn = nn.GRU(rnn_input_size, hidden_size)

    def get_overlap_inputs(self, inputs, overlap_size):
        overlap_inputs = []
        for rep in range(overlap_size):
            shift_inputs = inputs.roll(rep, dims=1)
            # pad sequence with 0
            shift_inputs[:, :rep, :] = 0
            overlap_inputs.append(shift_inputs)
        return torch.cat(overlap_inputs, dim=2)

    def forward(self, inputs):
        """
        :param inputs: Input data of shape (batch_size, num_nodes, num_timesteps, num_features).
        """
        inputs = self.get_overlap_inputs(
            inputs, overlap_size=self.overlap_size)

        encode_inputs = inputs

        inputs = inputs.permute(1, 0, 2)
        encode_inputs = encode_inputs.permute(1, 0, 2)

        out, _ = self.rnn(encode_inputs)

        # extract last input of encoder, used for decoder
        # 0 indicates target dim
        last = inputs.view(inputs.size(0), inputs.size(1),
                           self.overlap_size, self.num_features
                           )[-1, :, :, 0]

        return out, last.detach()


class Decoder(nn.Module):
    def __init__(self, num_features, num_timesteps_output, hidden_size, overlap_size):
        super(Decoder, self).__init__()

        self.overlap_size = overlap_size

        self.num_timesteps_output = num_timesteps_output

        rnn_input_size = overlap_size

        self.rnn_cell = nn.GRUCell(rnn_input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, encoder_out, encoder_hid, last):
        '''
        :param encoder_out: (num_timesteps_input, batch_size, hidden_size)
        :param encoder_hid: (batch_size, hidden_size)
        :param last: shape (batch_size, overlap_size)
        '''
        decoder_out = []

        hidden = encoder_hid

        for step in range(self.num_timesteps_output):
            attn_w = torch.einsum('ijk,jk->ij', encoder_out, hidden)
            attn_w = F.softmax(attn_w, dim=0)
            context = torch.einsum('ijk,ij->jk', encoder_out, attn_w)

            hidden = hidden + context

            decode_last = last

            hidden = self.rnn_cell(decode_last, hidden)

            out = self.linear(hidden)
            decoder_out.append(out)

            # roll last value
            last = torch.cat(
                [out.detach(), last[:, :self.overlap_size - 1]], dim=-1
            )

        decoder_out = torch.cat(decoder_out, dim=-1)
        return decoder_out


class KSeq2Seq(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps_input, num_timesteps_output, hidden_size, overlap_size, parallel):
        super(KSeq2Seq, self).__init__()

        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.hidden_size = hidden_size

        self.encoder_list = nn.ModuleList()
        for rep in range(parallel):
            self.encoder_list.append(
                Encoder(num_features, num_timesteps_input,
                        hidden_size, overlap_size
                        )
            )

        self.attn = nn.Embedding(num_nodes, parallel)

        if num_timesteps_output is not None:
            self.decoder = Decoder(
                num_features, num_timesteps_output, hidden_size, overlap_size
            )
        else:
            self.decoder = None

    def forward(self, X, n_id):
        # reshape to (batch_size * num_nodes, num_timesteps_input, num_features)
        num_nodes = X.size(1)
        inputs = X.reshape(-1, X.size(2), X.size(3))

        outs = []

        for idx in range(len(self.encoder_list)):
            out, last = self.encoder_list[idx](inputs)
            out = out.permute(1, 0, 2)
            out = out.view(-1, num_nodes,
                           self.num_timesteps_input, self.hidden_size)

            outs.append(out.unsqueeze(dim=-1))

        outs = torch.cat(outs, dim=-1)

        attn = self.attn(n_id)
        attn = torch.softmax(attn, dim=-1)

        encoder_out = torch.einsum('ijklm,jm->ijkl', outs, attn)

        _encoder_out = encoder_out

        if self.decoder is None:
            return _encoder_out, None
        else:
            encoder_out = encoder_out.reshape(
                -1, encoder_out.size(2), encoder_out.size(3)
            )
            encoder_out = encoder_out.permute(1, 0, 2)

            encoder_hid = encoder_out[-1]

            decoder_out = self.decoder(encoder_out, encoder_hid, last)
            _decoder_out = decoder_out.view(-1, X.size(1), self.num_timesteps_output)

            return _encoder_out, _decoder_out


class KRNN(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, hidden_size=64, overlap_size=3, parallel=10):
        super(KRNN, self).__init__()
        # set num_timesteps_output as None for only return encoder output
        self.seq2seq = KSeq2Seq(num_nodes, num_features, num_timesteps_input, num_timesteps_output,
                                hidden_size, overlap_size, parallel)

    def forward(self, X, n_id):
        return self.seq2seq(X, n_id)
