import torch
import torch.nn as nn
import math

class LoRALinearLayer(nn.Module):
    def __init__(self, Linear, rank, alpha):
        #linear 是原本线性层
        super().__init__()
        self.Linear = Linear
        self.in_feature = Linear.in_features
        self.out_feature = Linear.out_features

        self.rank = rank
        self.alpha = alpha
        self.scaling = self.alpha / self.rank
         
        # Parameter创建两个新的参数   Δw = B@A Δw@x = x@(A.T@B.T) 
        self.lora_A = nn.Parameter(torch.empty(self.rank,self.in_feature))
        self.lora_B = nn.Parameter(torch.empty(self.out_feature,self.rank))
        
        #初始化参数
        nn.init.kaiming_uniform_(self.lora_A,a = math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        #冻结Linear
        self.Linear.weight.requires_grad = False
        if self.Linear.bias is not None:
            self.Linear.bias.requires_grad = False
    
    def forward(self, x):
            Linear_output = self.Linear(x)
            return Linear_output + x @ ((self.lora_B @ self.lora_A).T * self.scaling)

