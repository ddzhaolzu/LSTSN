import torch.nn as nn
import torch
from torch.autograd import Variable
import math, copy
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class PositionalEncoding(nn.Module):
    "Implement the PE function."
    #def __init__(self, ft_size, time_len, domain):
    def __init__(self, ft_size, time_len, joint_num ):
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len

        # temporal positial embedding
        Tpos_list = list(range(self.time_len))  # 行向量

        # spatial positial embedding
        Spos_list = []

        for j_id in range(self.joint_num):
            Spos_list.append(j_id)

        S = torch.from_numpy(np.array(Spos_list)).unsqueeze(1).float()
        T = torch.from_numpy(np.array(Tpos_list)).unsqueeze(1).float()

        Spe = torch.zeros(self.joint_num, ft_size)  #[n, c]
        Tpe = torch.zeros(self.time_len, ft_size)  #[T, c]

        div_term = torch.exp(torch.arange(0, ft_size, 2).float() *
                             -(math.log(10000.0) / ft_size))

        Spe[:, 0::2] = torch.sin(S* div_term)
        Spe[:, 1::2] = torch.cos(S * div_term)if ft_size % 2 == 0 else torch.cos(S * div_term[:-1])

        Spe = Spe.unsqueeze(0).to(device)
        Spe = Spe.unsqueeze(0).to(device)

        Tpe[:, 0::2] = torch.sin(T * div_term)
        Tpe[:, 1::2] = torch.cos(T * div_term)if ft_size % 2 == 0 else torch.cos(T * div_term[:-1])
        Tpe = Tpe.unsqueeze(0).to(device)
        Tpe = Tpe.unsqueeze(2).to(device)


        self.register_buffer('Spe', Spe)
        self.register_buffer('Tpe', Tpe)
    def forward(self,x):
        # [batch_size, time_len, joint_num, 128]
        x = x + self.Spe + self.Tpe  
        return  x

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, ft_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(ft_dim))
        self.b_2 = nn.Parameter(torch.zeros(ft_dim))
        self.eps = eps

    def forward(self, x):
        #[batch, time, ft_dim)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2



class MultiHeadedAttention(nn.Module):
    def __init__(self, h_num, h_dim, input_dim, dp_rate, joint_num):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        #assert d_model % h == 0
        # We assume d_v always equals d_k
        self.h_dim = h_dim # head dimension
        self.h_num = h_num #head num
        self.joint_num = joint_num
        self.time_len = 8

        self.register_buffer('mask', self.get_domain_mask())


        # [batch_size, time_len-2, 3*joint_num, 8*32]
        self.key_map = nn.Sequential(
                            nn.Linear(input_dim, self.h_dim * self.h_num),
                            nn.Dropout(dp_rate),
                            )

        self.query_map = nn.Sequential(
                            nn.Linear(input_dim, self.h_dim * self.h_num),
                            nn.Dropout(dp_rate),
                            )

        self.value_map = nn.Sequential(
                            nn.Linear(input_dim, self.h_dim * self.h_num),
                            nn.ReLU(),
                            nn.Dropout(dp_rate),
                                     )
        self.cls = nn.Linear(3*self.joint_num, self.joint_num)

    def get_domain_mask(self):
        # Sec 3.4
        mask = torch.ones(3 * self.joint_num, 3 * self.joint_num)
        filted_area = torch.zeros(self.joint_num, self.joint_num)

        mask[0:self.joint_num, self.joint_num * 2:self.joint_num * 3] *= filted_area
        mask[self.joint_num * 2:self.joint_num * 3, 0:self.joint_num] *= filted_area

        mask = Variable((mask)).to(device)  ####################################cuda()
        
        
        return mask



    def attention(self,query, key, value):
        "Compute 'Scaled Dot Product Attention'"
        # [batch, time, ft_dim)
        d_k = query.size(-1)  #
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)    # W  #[batch_size, time_len-2, 8, 3*joint_num, 3*joint_num]

        scores *= self.mask


        if self.joint_num==22:

            m=torch.max(scores,-1)[0]
            m=torch.unsqueeze(m,axis=-1)    
            scores [torch.where(abs(scores)<=m/9)]= (-9e15)#'''
            #scores[torch.where(abs(scores) <= 0.05)] = (-9e15)
        else:
            #scores += (1 - self.mask) * (-9e15)
            scores[torch.where(abs(scores) ==0)] = (-9e15)
        scores = F.softmax(scores, dim=-1)

        return torch.matmul(scores, value), scores

    def forward(self, x):
        "Implements Figure 2"
        nbatches = x.size(0)  # # [batch_size, time_len-2, 3*joint_num, 128]

        # # [batch_size, time_len-2, 3*joint_num, 8*32]--> [batch_size, time_len-2, 3*joint_num, 8, 32] --> [batch_size, time_len-2, 8, 3*joint_num, 32]
        query = self.query_map(x).view(nbatches, self.time_len-2, 3*self.joint_num, self.h_num, self.h_dim).transpose(2, 3)  #swap dimension 2 and 3
        key = self.key_map(x).view(nbatches, self.time_len-2, 3*self.joint_num, self.h_num, self.h_dim).transpose(2, 3)
        value = self.value_map(x).view(nbatches, self.time_len-2, 3*self.joint_num, self.h_num, self.h_dim).transpose(2, 3)


        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value) ## [batch_size, time_len-2, 8, 3*joint_num, 32]
        x = x.transpose(3, 4) # [batch_size, time_len-2, 8, 32, 3*joint_num]
        x = self.cls(x) # [batch_size, time_len-2, 8, 32, joint_num]

        x = x.transpose(3, 4) # [batch_size, time_len-2, 8, joint_num ,  32]

        # 3) "Concat" using a view and apply a final linear.
        # [batch_size, time_len-2, 8, joint_num ,  32]---># [batch_size, time_len-2, joint_num , 8,  32]--->[batch_size, time_len-2, joint_num , 8*32]
        x = x.transpose(2, 3).contiguous() \
            .view(nbatches, self.time_len-2, self.joint_num, self.h_dim * self.h_num)   # [batch_size, time_len-2, joint_num , 8*32]         [batch, N, h_dim * h_num ]
        x = x.contiguous() \
            .view(nbatches, (self.time_len-2)*self.joint_num, self.h_dim * self.h_num)  # [batch_size, (time_len-2)*joint_num, 8*32]
        return x

class ST_ATT_Layer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, input_size, output_size, h_num, h_dim, dp_rate, time_len, joint_num ):
        #input_size : the dim of input
        #output_size: the dim of output
        #h_num: att head num
        #h_dim: dim of each att head
        #time_len: input frame number
        #domain: do att on spatial domain or temporal domain

        super(ST_ATT_Layer, self).__init__()

        self.pe = PositionalEncoding(input_size, time_len,joint_num )

        #h_num, h_dim, input_dim, dp_rate,domain
        self.attn = MultiHeadedAttention(h_num, h_dim, input_size, dp_rate, joint_num ) #do att on input dim

        self.ft_map = nn.Sequential(
                        nn.Linear(h_num * h_dim, output_size),
                        nn.ReLU(),
                        LayerNorm(output_size),
                        nn.Dropout(dp_rate),
                        )
        self.init_parameters()

    def forward(self, x):
        # # [b,8,6,128]  f
        b = x.shape[0]
        time_len = x.shape[1]
        joint_num = x.shape[2]

        x = self.pe(x)  # # [batch_size, time_len, joint_num, 128]
        X = torch.zeros((b, time_len - 2, 3, joint_num, 128))
        for i in range(time_len - 2):
            X[:, i, :, :, :] = x[:,  i:(i + 3), :, :]    # [batch_size, time_len-2, 3, joint_num, 128]

        x = X.to(device)
        # reshape x
        x = x.reshape(-1, time_len-2, 3 * joint_num, 128)  # [batch_size, time_len-2, 3*joint_num, 128]

        x = self.attn(x)     ## [batch_size, (time_len-2)*joint_num, 8*32]
        x = self.ft_map(x) # [batch_size, (time_len-2)*joint_num, 128]
        return x

    def init_parameters(self):
        model_list = [ self.attn, self.ft_map]
        for model in model_list:
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)