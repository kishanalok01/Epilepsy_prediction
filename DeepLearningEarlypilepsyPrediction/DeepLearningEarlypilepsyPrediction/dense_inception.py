import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from .basic_conv import *
from .inceptions import Inception4

class DenseInception(nn.Module):
    def __init__(self, params, data_shape, in_channels):
        super(DenseInception, self).__init__()
        self.num_channels = params.num_channels
        self.data_shape = data_shape
        self.num_inception_layers = params.num_inception_layers            
        self.inception_0 = Inception4(1, pool_features = self.num_channels, filter_size = [9, 15, 21], pool_size = 5)
#         self.conv1x1 = nn.ModuleList([])
        self.inception_1 = Inception4(self.num_channels * 3, pool_features = self.num_channels*3, filter_size = [9, 13, 17], pool_size = 4)
        self.conv1x1_10 = BasicConv2d(self.num_channels * 12, self.num_channels*9, False, kernel_size = (1,1), stride = 1)

        self.inception_2 = Inception4(self.num_channels *9 , pool_features = self.num_channels*9, filter_size = [7,11,15], pool_size = 4)
        
        
        self.conv1x1_2 = BasicConv2d(self.num_channels * 27, self.num_channels*18, False, kernel_size = (1,1), stride = 1)
        
        self.inception_3 = Inception4(self.num_channels * 18, pool_features = self.num_channels*18, filter_size = [5, 7, 9], pool_size = 3)
        
        self.conv1x1_3 = BasicConv2d(self.num_channels * 54, self.num_channels*18, False, kernel_size = (1,1), stride = 1)
        
        self.conv1x1_32 = BasicConv2d(self.num_channels * 36, self.num_channels*18, False, kernel_size = (1,1), stride = 1)
        
        self.inception_4 = Inception4(self.num_channels * 18, pool_features = self.num_channels*18, filter_size = [3, 5, 7], pool_size = 3)
        
        self.conv1x1_4 = BasicConv2d(self.num_channels*54, self.num_channels*18, False, kernel_size = (1,1), stride = 1)
        
        self.inception_5 = Inception4(self.num_channels * 18, pool_features = self.num_channels*18, filter_size = [3, 5, 7], pool_size = 3)
        
        self.conv1x1_5 = BasicConv2d(self.num_channels*54, self.num_channels*27, False, kernel_size = (1,1), stride = 1)
        
        self.conv1x1_54 = BasicConv2d(self.num_channels*45, self.num_channels*18, False, kernel_size = (1,1), stride = 1)
        
        ##one more!
        
        self.inception_6 = Inception4(self.num_channels * 18, pool_features = self.num_channels*18, filter_size = [3, 5, 7], pool_size = 3)
        
        self.conv1x1_6 = BasicConv2d(self.num_channels*54, self.num_channels*18, False, kernel_size = (1,1), stride = 1)
        
        self.inception_7 = Inception4(self.num_channels * 18, pool_features = self.num_channels*18, filter_size = [3, 5, 7], pool_size = 3)
        
        self.conv1x1_7 = BasicConv2d(self.num_channels*54, self.num_channels*27, False, kernel_size = (1,1), stride = 1)
        
        self.conv1x1_76 = BasicConv2d(self.num_channels*45, self.num_channels*36, False, kernel_size = (1,1), stride = 1)
                
        self.fc1 = nn.Linear(self.data_shape[1] * self.num_channels * 36 * int(self.data_shape[0] / (7*5*5*4)), 128) 
        self.fcbn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 1) 
        self.dropout_rate = params.dropout_rate   
        print("now the dropout_rate", params.dropout_rate)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                
    def forward(self, s):
        s = s.unsqueeze(1) #32 x 1 x len x num_channels
#         for i in range(self.num_inception_layers):
#             s = self.inceptions[i](s)
#             if(i>0):
#                 s = self.conv1x1[i-1](s)
        s_0 = self.inception_0(s)
        s_1 = self.inception_1(s_0)
        s_cat_10 = torch.cat([s_0,s_1], 1)
        s = self.conv1x1_10(s_cat_10)
        s = F.max_pool2d(s, (7,1))
        s_0 = self.inception_2(s)
        s_0 = self.conv1x1_2(s_0)
        s_1 = self.inception_3(s_0)
        s_1 = self.conv1x1_3(s_1)
        s_cat_10 = torch.cat([s_0,s_1],1)
        s = self.conv1x1_32(s_cat_10)
        s = F.max_pool2d(s, (5,1))
        
        s_0 = self.inception_4(s)        
        s_0 = self.conv1x1_4(s_0)
        s_1 = self.inception_4(s_0)
        s_1 = self.conv1x1_5(s_1)
        s_cat_10 = torch.cat([s_0,s_1],1)
        s = self.conv1x1_54(s_cat_10)
        s = F.max_pool2d(s, (5,1))
        
        s_0 = self.inception_6(s)        
        s_0 = self.conv1x1_6(s_0)
        s_1 = self.inception_6(s_0)
        s_1 = self.conv1x1_7(s_1)
        s_cat_10 = torch.cat([s_0,s_1],1)
        s = self.conv1x1_76(s_cat_10)
#         print(s.size())
        s = F.max_pool2d(s, (4,1))        
        
#         print(s.size())
        s = s.contiguous()
        s = s.view(s.size()[0], -1)
        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))),p=self.dropout_rate, training=self.training)
        s = self.fc2(s)
#         print(F.sigmoid(s))
        return F.sigmoid(s)         
        
        
#         s = s.unsqueeze(1) #32 x 1 x len x num_channels
# #         for i in range(self.num_inception_layers):
# #             s = self.inceptions[i](s)
# #             if(i>0):
# #                 s = self.conv1x1[i-1](s)
#         s_0 = self.inception_0(s)
#         s_1 = self.inception_1(s_0)

#         s_cat_10 = torch.cat([s_0,s_1], 1)
#         s = self.conv1x1_10(s_cat_10)
#         s = F.max_pool2d(s, (5,1))
#         print(s.size())
#         s_2 = self.inception_2(s)
# #         print("s_2", s_2.size())
#         s_2 = self.conv1x1_2(s_2)
#         s_3 = self.inception_3(s_2)
#         s_3 = self.conv1x1_3(s_3)
#         s_cat_32 = torch.cat([s_3,s_2],1)
#         s = self.conv1x1_32(s_cat_32)
#         s = F.max_pool2d(s, (5,1))

#         s_4 = self.inception_4(s)        
#         s_4 = self.conv1x1_4(s_4)
#         s_5 = self.inception_4(s_4)
#         s_5 = self.conv1x1_5(s_5)
#         s_cat_54 = torch.cat([s_5,s_4],1)
#         s = F.max_pool2d(s_cat_54, (5,1))
#         print(s.size())
#         s = s.contiguous()
#         s = s.view(s.size()[0], -1)
#         s = F.dropout(F.relu(self.fcbn1(self.fc1(s))),p=self.dropout_rate, training=self.training)
#         s = self.fc2(s)
#         return F.sigmoid(s) 




# class DenseInception(nn.Module):
#     def __init__(self, params, data_shape, in_channels):
#         super(DenseInception, self).__init__()
#         self.num_channels = params.num_channels
#         self.data_shape = data_shape
#         self.num_inception_layers = params.num_inception_layers            
#         self.inceptions = nn.ModuleList([Inception3(1, pool_features = self.num_channels, filter_size = [9, 15, 21], pool_size = 5)])
#         self.conv1x1 = nn.ModuleList([])
#         self.inceptions.append(Inception3(self.num_channels * 3, pool_features = self.num_channels*3, filter_size = [9, 13, 17], pool_size = 4))
#         self.conv1x1.append(BasicConv2d(self.num_channels*9, self.num_channels*6, False, kernel_size = (1,1), stride = 1))

#         self.inceptions.append(Inception3(self.num_channels * 6, pool_features = self.num_channels*6, filter_size = [7,11,15], pool_size = 4))  
        
#         self.conv1x1.append(BasicConv2d(self.num_channels * 18, self.num_channels*12, False, kernel_size = (1,1), stride = 1))
        
#         self.inceptions.append(Inception3(self.num_channels * 12, pool_features = self.num_channels*12, filter_size = [5, 7, 9], pool_size = 3))
#         self.conv1x1.append(BasicConv2d(self.num_channels*36, self.num_channels*24, False, kernel_size = (1,1), stride = 1))
        
#         self.inceptions.append(Inception3(self.num_channels * 24, pool_features = self.num_channels*24, filter_size = [3, 5, 7], pool_size = 3))
        
#         self.conv1x1.append(BasicConv2d(self.num_channels*72, self.num_channels*36, False, kernel_size = (1,1), stride = 1))
        
        
        
        
#         self.fc1 = nn.Linear(self.data_shape[1] * self.num_channels * 36 * int(self.data_shape[0] / (5*4*4*3*3)), 400)       

#         self.fcbn1 = nn.BatchNorm1d(400)
#         self.fc2 = nn.Linear(400, 1) 
#         self.dropout_rate = params.dropout_rate   
        
#     def forward(self, s):
#         s = s.unsqueeze(1) #32 x 1 x len x num_channels
#         for i in range(self.num_inception_layers):
#             s = self.inceptions[i](s)
#             if(i>0):
#                 s = self.conv1x1[i-1](s)
#         s = s.contiguous()
#         s = s.view(s.size()[0], -1)
#         s = F.dropout(F.relu(self.fcbn1(self.fc1(s))),p=self.dropout_rate, training=self.training)
#         s = self.fc2(s)
#         return F.sigmoid(s) 
    
