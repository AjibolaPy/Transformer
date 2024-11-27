import torch 
import numpy as np
import torch.nn as nn


class QWENVLM(nn.Module):

    def __init__(self, ndim:int, reduced):
        super().__init__()
        """
        THIS THE FEED FORWARD LAYER FOR DIMENSIONAL REDUCTION USES GLU ACTIVATION.
        
        ndim: the initial number of neuroon
        reduced : the output number of neuron 
        """
        print(ndim, ndim*2)
        self.linear1=nn.Linear(ndim, ndim*2)
        self.linear2=nn.Linear(ndim, reduced)
        self.layer_norm1=nn.LayerNorm(ndim)
        self.layer_norm2=nn.LayerNorm(reduced)
    def forward(self, x):
        normalized=self.layer_norm1(x)
        print("passed here", normalized.shape)
        x1=self.linear1(normalized)
        x2=nn.GLU(dim=-1)(x1)
        print("After GLU LAYER ", x2.shape)
        x3=self.linear2(x2)
        
        return x3
      #  return self.layer_norm2(nn.GLU()(x3)) 

