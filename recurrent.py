import torch
from linear import linear_fun
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F


def rec_lin_func(input, hidden, w_hh, w_ih, b_hh, b_ih):
    output = input.matmul(w_ih.t()) + hidden.matmul(w_hh)
    if b_hh is not None:
        output += b_hh
    if b_ih is not None:
        output += b_ih

    return F.tanh(output)


class Rec(nn.Module):
    def __init__(self, input_size, hidden_size, bias=None):
        super(Rec, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.w_ih = Parameter(torch.Tensor(hidden_size, input_size))
        self.w_hh = Parameter(torch.Tensor(hidden_size, hidden_size))

        if self.bias:
            self.b_hh = Parameter(torch.Tensor(self.hidden_size))
            self.b_ih = Parameter(torch.Tensor(self.hidden_size))
        else:
            self.b_hh = None
            self.b_ih = None

    def init_params(self):
        torch.nn.init.xavier_uniform(self.w_ih)
        torch.nn.init.xavier_uniform(self.w_hh)
        if self.bias:
            torch.nn.init.xavier_uniform(self.b_ih)
            torch.nn.init.xavier_uniform(self.b_hh)

    def forward(self, input, hidden):
        return rec_lin_func(input, hidden, self.w_hh, self.w_ih, self.b_hh, self.b_ih)
