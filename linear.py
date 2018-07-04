import torch 
import torch.nn as nn
from torch.nn.parameter import Parameter


def linear_fun(inpt: torch.Tensor, wt: torch.Tensor, bias: torch.Tensor= None):
    out = inpt.matmul(wt.t())
    if bias is not None:
        return out + bias


class Linear(nn.Module):
    def __init__(self, input_size, output_size, bias=None):
        super(Linear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.bias = bias
        self.w = Parameter(torch.Tensor(self.output_size, self.input_size))
        if self.bias:
            self.b = Parameter(torch.Tensor(self.output_size))
        else:
            self.b = None

    def init_params(self):
        torch.nn.init.xavier_uniform(self.w)
        if self.bias:
            torch.nn.init.xavier_uniform(self.b)

    def forward(self, input):
        return linear_fun(input, self.w, self.b)



