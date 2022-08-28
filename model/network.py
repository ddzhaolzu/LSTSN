from .Local_st_att_layerA import *
import torch.nn as nn
#import torch

class DG_STA(nn.Module):     ####初始化model
    def __init__(self, num_classes, dp_rate):
        super(DG_STA, self).__init__()

        h_dim = 32
        h_num= 8

        #'''torch.nn.Sequential是一个Sequential容器，模块将按照构造函数中传递的顺序添加到模块中。另外，也可以传入一个有序模块。'''
        #nn.Linear(in_features = 64*64*3, out_features = 1)
        #nn.Linear（）是用于设置网络中的全连接层的
        self.input_map = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            LayerNorm(128),  #层归一化
            nn.Dropout(dp_rate),
        )

        '''self.input_mapv = nn.Sequential(
            nn.Linear(96, 128),
            nn.ReLU(),
            LayerNorm(128),  # 层归一化
            nn.Dropout(dp_rate),
        )'''


        #input_size, h_num, h_dim, dp_rate, time_len, domain
        self.atts = ST_ATT_Layer(input_size=128,output_size= 128, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, time_len = 8, joint_num =22)
        self.attv = ST_ATT_Layer(input_size=128, output_size=128, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, time_len=8,joint_num=4)
        self.attf = ST_ATT_Layer(input_size=128, output_size=128, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, time_len=8,joint_num=6)


        #self.attsv = ST_ATT_Layer(input_size=128, output_size=128, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, time_len=8, joint_num=26)
        #self.attsf = ST_ATT_Layer(input_size=128, output_size=128, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, time_len=8, joint_num=28)
        #self.attsfv = ST_ATT_Layer(input_size=128, output_size=128, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, time_len=8, joint_num=32)



        self.cls = nn.Linear(128, num_classes)



    def forward(self, x,f,v):
        # input shape: [batch_size,time_len, joint_num, 3]

        #input map

       ### [batch_size, time_len, joint_num, 128]
        x = self.input_map(x)  # [b,8,joint_num,128]
        f = self.input_map(f)
        v = self.input_map(v)

        x = self.atts(x)## [batch_size, (time_len-2)*joint_num, 128]
        v = self.attv(v)  ## [batch_size, (time_len-2)*4, 128]
        f = self.attf(f)  ## [batch_size, (time_len-2)*6, 128]

        ####张量拼接
        x = torch.cat((x, v), 1)  ##[b, (time_len-2)*joint_num+8*4 , 128]


        x = x.sum(1) /x.shape[1]  # [batch_size, 128]

        pred = self.cls(x)  # [batch_size, 14/28]
        return pred