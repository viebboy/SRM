import math
import Layers
import torch
import torch.nn as nn
import torch.nn.functional as F



class AllCNN(nn.Module):
    def __init__(self, scale, nb_class=10):
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #print(x.size())
        x = self.block1(x)
        #print(x.size())
        x = self.block2(x)
        #print(x.size())
        x = self.block3(x)
        #print(x.size())
        x = self.block4(x)
        #print(x.size())
        x = F.avg_pool2d(x, 7)
        #print(x.size())
        x = x.view(x.size(0), -1)
        x = self.output_layer(x)

        return x 



class PredAllCNN(nn.Module):
    def __init__(self, nb_codewords,
                       scale, 
                       nb_class=10,
                       return_output=False):
        super(PredAllCNN, self).__init__()
       
        self.scale = scale
        in_dims = []
        self.return_output = return_output

        nb_conv = int(48*scale)
        self.conv1 = nn.Conv2d(3, nb_conv, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(nb_conv)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
       
        in_dims.append(nb_conv)

        # 56x56x48 
        in_conv = 48
        nb_conv = int(96*scale)
        self.block1 = nn.Sequential(*self.block_(((in_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 2, 1))))

        in_dims.append(nb_conv)

        # 28x28x96 
        in_conv = nb_conv
        nb_conv = int(192*scale)
        self.block2 = nn.Sequential(*self.block_(((in_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 2, 1))))
                 
        in_dims.append(nb_conv)

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
        
        in_dims.append(nb_conv)
        in_dims.append(nb_conv)

        self.output_layer = nn.Linear(nb_conv, nb_class)

        
        self.sparse_layers = nn.ModuleList()

        for idx, in_dim in enumerate(in_dims):
            self.sparse_layers.append(Layers.GlobalLocalPred(in_dim,
                                                           nb_codewords[idx]))


        
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

    def forward(self, x):
        if self.return_output:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = F.avg_pool2d(x, 7)
            x = x.view(x.size(0), -1)
            x = self.output_layer(x)
            return x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        gl0, ll0 = self.sparse_layers[0](x)

        x = self.block1(x)
        gl1, ll1 = self.sparse_layers[1](x)
        x = self.block2(x)
        gl2, ll2 = self.sparse_layers[2](x)
        x = self.block3(x)
        x = self.block4(x)
        gl3, ll3 = self.sparse_layers[3](x)
        x = F.avg_pool2d(x, 7)
        gl4, ll4 = self.sparse_layers[4](x)
        
        return (gl0, gl1, gl2, gl3, gl4),\
                (ll0, ll1, ll2, ll3, ll4)



class HintPredAllCNN(nn.Module):
    def __init__(self, hint_dims,
                       scale,
                       nb_class=10,
                       return_output=False):
        super(HintPredAllCNN, self).__init__()
       
        self.scale = scale
        in_dims = []
        self.return_output = return_output

        nb_conv = int(48*scale)
        self.conv1 = nn.Conv2d(3, nb_conv, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(nb_conv)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
       
        in_dims.append(nb_conv)

        # 56x56x48 
        in_conv = 48
        nb_conv = int(96*scale)
        self.block1 = nn.Sequential(*self.block_(((in_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 2, 1))))

        in_dims.append(nb_conv)

        # 28x28x96 
        in_conv = nb_conv
        nb_conv = int(192*scale)
        self.block2 = nn.Sequential(*self.block_(((in_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 2, 1))))
                 
        in_dims.append(nb_conv)

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
        
        in_dims.append(nb_conv)
        in_dims.append(nb_conv)

        self.output_layer = nn.Linear(nb_conv, nb_class)

        
        self.hint_layers = nn.ModuleList()

        for idx, in_dim in enumerate(in_dims):
            self.hint_layers.append(nn.Conv2d(in_channels=in_dim,
                                              out_channels=hint_dims[idx],
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))


        
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

    def forward(self, x):
        if self.return_output:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = F.avg_pool2d(x, 7)
            x = x.view(x.size(0), -1)
            x = self.output_layer(x)
            return x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        hint0 = self.hint_layers[0](x)

        x = self.block1(x)
        hint1 = self.hint_layers[1](x)
        x = self.block2(x)
        hint2 = self.hint_layers[2](x)
        x = self.block3(x)
        x = self.block4(x)
        hint3 = self.hint_layers[3](x)
        x = F.avg_pool2d(x, 7)
        hint4 = self.hint_layers[4](x)
        
        return hint0, hint1, hint2, hint3, hint4 

