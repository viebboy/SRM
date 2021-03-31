import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class AllCNN(nn.Module):
    def __init__(self, scale=0.66, nb_class=10):
        super(AllCNN, self).__init__()
       
        self.scale = scale
       
        in_conv = 3
        nb_conv = int(96*scale)
        self.block1 = nn.Sequential(*self.block_(((in_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 2, 1))))

       
        in_conv = nb_conv
        nb_conv = int(192*scale)
        self.block2 = nn.Sequential(*self.block_(((in_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 2, 1))))
                    
        in_conv = nb_conv
        nb_conv = int(256*scale)
        self.block3 = nn.Sequential(*self.block_(((in_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 2, 1))))
                
       
        in_conv = nb_conv
        nb_conv = int(384*scale)
        self.block4 = nn.Sequential(*self.block_(((in_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 1, 1, 0))))
        
        self.output_layer = nn.Linear(nb_conv, nb_class)

        self.LeakyReLU = nn.LeakyReLU(0.1)

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


        layers.pop(-1)
       
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
        feat_m.append(self.block1)
        feat_m.append(self.LeakyReLU)
        feat_m.append(self.block2)
        feat_m.append(self.LeakyReLU)
        feat_m.append(self.block3)
        feat_m.append(self.LeakyReLU)
        feat_m.append(self.block4)
        feat_m.append(self.LeakyReLU)
        return feat_m

    def forward(self, x, is_feat=False, preact=False):
       
        x = self.block1(x)
        f0_preact = x
        x = self.LeakyReLU(x)
        f0 = x 
        
        x = self.block2(x)
        f1_preact = x
        x = self.LeakyReLU(x)
        f1 = x
        
        x = self.block3(x)
        x = self.LeakyReLU(x)
        
        x = self.block4(x)
        f2_preact = x
        x = self.LeakyReLU(x)
        f2 = x

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)

        f3 = x
        f3_preact = x
        x = self.output_layer(x)

        if is_feat:
            if preact:
                return [f0_preact, f1_preact, f2_preact, f3_preact], x
            else:
                return [f0, f1, f2, f3], x
        else:
            return x 

