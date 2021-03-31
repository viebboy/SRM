import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class AllCNN(nn.Module):
    def __init__(self, scale=1.0, nb_class=10):
        super(AllCNN, self).__init__()
       
        self.scale = scale
      
        nb_conv = int(48*scale)
        self.conv1 = nn.Conv2d(3, nb_conv, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(nb_conv)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        
        # 56x56x48 
        in_conv = 48
        nb_conv = int(96*scale)
        self.block1 = nn.Sequential(*self.block_(((in_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 2, 1))))

        # 28x28x96 
        in_conv = nb_conv
        nb_conv = int(192*scale)
        self.block2 = nn.Sequential(*self.block_(((in_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 2, 1))))
                 
        # 14x14x192
        in_conv = nb_conv
        nb_conv = int(256*scale)
        self.block3 = nn.Sequential(*self.block_(((in_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 2, 1))))
                
        # 7x7x 256 
        in_conv = nb_conv
        nb_conv = int(384*scale)
        self.block4 = nn.Sequential(*self.block_(((in_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 1, 1, 0))))
        
        self.output_layer = nn.Linear(nb_conv, nb_class)

        self.initialize()

    def block_(self, topology):
       
        layers = []
        for b_ in topology:

            in_, out_, kernel, stride, pad = b_ 

            layers.append(nn.Conv2d(in_channels=in_,
                                         out_channels=out_,
                                         kernel_size=kernel,
                                         stride=stride,
                                         padding=pad))

            layers.append(nn.BatchNorm2d(out_))
            layers.append(nn.LeakyReLU(0.1))
       
        return layers
    
    def initialize(self,):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.normal_(layer.bias)
            
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.normal_(layer.bias)

            if isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1.0)
                nn.init.constant_(layer.bias, 0.0)


    def get_feat_modules(self,):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.relu)
        feat_m.append(self.maxpool)
        feat_m.append(self.block1)
        feat_m.append(self.block2)
        feat_m.append(self.block3)
        feat_m.append(self.block4)

        return feat_m



    def forward(self, x, is_feat=False, preact=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        f0 = x
        
        x = self.block1(x)
        f1 = x
        x = self.block2(x)
        f2 = x
        x = self.block3(x)
        x = self.block4(x)
        f3 = x
        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        f4 = x
        x = self.output_layer(x)

        if is_feat:
            return [f0, f1, f2, f3, f4], x
        else:
            return x 


