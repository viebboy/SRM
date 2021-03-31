import math
import Layers
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalLocalPred(nn.Module):
    def __init__(self, 
            codeword_multiplier,
            centers,
            intercept,
            scale, 
            nb_class=10,
            return_output=False):
        
        super(GlobalLocalPred, self).__init__()
       
        self.scale = scale
        self.return_output = return_output 
        in_dims = []

        in_conv = 3
        nb_conv = int(96*scale)
        self.block1 = nn.Sequential(*self.block_(((in_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 2, 1))))

       
        in_dims.append(nb_conv)

        in_conv = nb_conv
        nb_conv = int(192*scale)
        self.block2 = nn.Sequential(*self.block_(((in_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 2, 1))))
                    
        in_dims.append(nb_conv)

        in_conv = nb_conv
        nb_conv = int(256*scale)
        self.block3= nn.Sequential(*self.block_(((in_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 2, 1))))
                
        in_conv = nb_conv
        nb_conv = int(384*scale)
        self.block4= nn.Sequential(*self.block_(((in_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 1, 1, 0))))

        in_dims.append(nb_conv)

        self.output_layer = nn.Linear(nb_conv, nb_class)

        # nbof layers
        if centers is None:
            centers = [None,]*3
        if intercept is None:
            intercept = [None,]*3

        self.sparse_layers = nn.ModuleList()
        
        teacher_in_dims = [128, 256, 1024]

        self.sparse_layers.append(Layers.GlobalLocalPred(in_dims[0],
                                                 teacher_in_dims[0]*codeword_multiplier,
                                                 centers=centers[0],
                                                 intercept=intercept[0]))

        self.sparse_layers.append(Layers.GlobalLocalPred(in_dims[1],
                                                 teacher_in_dims[1]*codeword_multiplier,
                                                 centers=centers[1],
                                                 intercept=intercept[1]))

        self.sparse_layers.append(Layers.GlobalLocalPred(in_dims[2],
                                                 teacher_in_dims[2]*codeword_multiplier,
                                                 centers=centers[2],
                                                 intercept=intercept[2]))

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
    
    def forward(self, x):
       
        out1 = self.block1(x)
        ghist1, lhist1 = self.sparse_layers[0](out1)

        out2 = self.block2(out1)
        ghist2, lhist2 = self.sparse_layers[1](out2)

        out3 = self.block3(out2)
        out4 = self.block4(out3)
        ghist3, lhist3 = self.sparse_layers[2](out4)
        
        x = F.avg_pool2d(out4, 4)
        x = x.view(x.size(0), -1)
        x = self.output_layer(x)

        if self.return_output:
            return x
        else:
            return (ghist1, ghist2, ghist3), (lhist1, lhist2, lhist3) 

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


class LocalPred(nn.Module):
    def __init__(self, 
            codeword_multiplier,
            centers,
            intercept,
            scale, 
            nb_class=10,
            return_output=False):
        
        super(LocalPred, self).__init__()
       
        self.scale = scale
        self.return_output = return_output 
        in_dims = []

        in_conv = 3
        nb_conv = int(96*scale)
        self.block1 = nn.Sequential(*self.block_(((in_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 2, 1))))

       
        in_dims.append(nb_conv)

        in_conv = nb_conv
        nb_conv = int(192*scale)
        self.block2 = nn.Sequential(*self.block_(((in_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 2, 1))))
                    
        in_dims.append(nb_conv)

        in_conv = nb_conv
        nb_conv = int(256*scale)
        self.block3= nn.Sequential(*self.block_(((in_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 2, 1))))
                
        in_conv = nb_conv
        nb_conv = int(384*scale)
        self.block4= nn.Sequential(*self.block_(((in_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 1, 1, 0))))

        in_dims.append(nb_conv)

        self.output_layer = nn.Linear(nb_conv, nb_class)

        # nbof layers
        if centers is None:
            centers = [None,]*3
        if intercept is None:
            intercept = [None,]*3

        self.sparse_layers = nn.ModuleList()
        teacher_in_dims = [128, 256, 1024]

        self.sparse_layers.append(Layers.LocalPred(in_dims[0],
                                                 teacher_in_dims[0]*codeword_multiplier,
                                                 centers=centers[0],
                                                 intercept=intercept[0]))

        self.sparse_layers.append(Layers.LocalPred(in_dims[1],
                                                 teacher_in_dims[1]*codeword_multiplier,
                                                 centers=centers[1],
                                                 intercept=intercept[1]))

        self.sparse_layers.append(Layers.LocalPred(in_dims[2],
                                                 teacher_in_dims[2]*codeword_multiplier,
                                                 centers=centers[2],
                                                 intercept=intercept[2]))

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
    
    def forward(self, x):
       
        out1 = self.block1(x)
        hist1 = self.sparse_layers[0](out1)

        out2 = self.block2(out1)
        hist2 = self.sparse_layers[1](out2)

        out3 = self.block3(out2)
        out4 = self.block4(out3)
        hist3 = self.sparse_layers[2](out4)
        
        x = F.avg_pool2d(out4, 4)
        x = x.view(x.size(0), -1)
        x = self.output_layer(x)

        if self.return_output:
            return x
        else:
            return hist1, hist2, hist3 

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




class GlobalPred(nn.Module):
    def __init__(self, 
            codeword_multiplier,
            centers,
            intercept,
            scale, 
            nb_class=10,
            return_output=False):
        
        super(GlobalPred, self).__init__()
       
        self.scale = scale
        self.return_output = return_output 
        in_dims = []

        in_conv = 3
        nb_conv = int(96*scale)
        self.block1 = nn.Sequential(*self.block_(((in_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 2, 1))))

       
        in_dims.append(nb_conv)

        in_conv = nb_conv
        nb_conv = int(192*scale)
        self.block2 = nn.Sequential(*self.block_(((in_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 2, 1))))
                    
        in_dims.append(nb_conv)

        in_conv = nb_conv
        nb_conv = int(256*scale)
        self.block3= nn.Sequential(*self.block_(((in_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 2, 1))))
                
        in_conv = nb_conv
        nb_conv = int(384*scale)
        self.block4= nn.Sequential(*self.block_(((in_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 1, 1, 0))))

        in_dims.append(nb_conv)

        self.output_layer = nn.Linear(nb_conv, nb_class)

        # nbof layers
        if centers is None:
            centers = [None,]*3
        if intercept is None:
            intercept = [None,]*3

        self.sparse_layers = nn.ModuleList()
        teacher_in_dims = [128, 256, 1024]

        self.sparse_layers.append(Layers.GlobalPred(in_dims[0],
                                                 teacher_in_dims[0]*codeword_multiplier,
                                                 centers=centers[0],
                                                 intercept=intercept[0]))

        self.sparse_layers.append(Layers.GlobalPred(in_dims[1],
                                                 teacher_in_dims[1]*codeword_multiplier,
                                                 centers=centers[1],
                                                 intercept=intercept[1]))

        self.sparse_layers.append(Layers.GlobalPred(in_dims[2],
                                                 teacher_in_dims[2]*codeword_multiplier,
                                                 centers=centers[2],
                                                 intercept=intercept[2]))

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
    
    def forward(self, x):
       
        out1 = self.block1(x)
        hist1 = self.sparse_layers[0](out1)

        out2 = self.block2(out1)
        hist2 = self.sparse_layers[1](out2)

        out3 = self.block3(out2)
        out4 = self.block4(out3)
        hist3 = self.sparse_layers[2](out4)
        
        x = F.avg_pool2d(out4, 4)
        x = x.view(x.size(0), -1)
        x = self.output_layer(x)

        if self.return_output:
            return x
        else:
            return hist1, hist2, hist3 

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



class HintAllCNN(nn.Module):
    def __init__(self, hint_dims, scale, nb_class=10, return_output=False):
        super(HintAllCNN, self).__init__()
       
        self.scale = scale
        self.hint_layers = nn.ModuleList() 
        self.return_output = return_output

        in_conv = 3
        nb_conv = int(96*scale)
        self.block1 = nn.Sequential(*self.block_(((in_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 2, 1))))

        self.hint_layers.append(nn.Conv2d(in_channels=nb_conv,
                                          out_channels=hint_dims[0],
                                          kernel_size=1,
                                          stride=1,
                                          padding=0))

        
        in_conv = nb_conv
        nb_conv = int(192*scale)
        self.block2 = nn.Sequential(*self.block_(((in_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 1, 1),
                     (nb_conv, nb_conv, 3, 2, 1))))
                    
        self.hint_layers.append(nn.Conv2d(in_channels=nb_conv,
                                          out_channels=hint_dims[1],
                                          kernel_size=1,
                                          stride=1,
                                          padding=0))
        
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
        
        self.hint_layers.append(nn.Conv2d(in_channels=nb_conv,
                                          out_channels=hint_dims[2],
                                          kernel_size=1,
                                          stride=1,
                                          padding=0)) 

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
    
    def forward(self, x):
       
        x = self.block1(x)
        hint1 = self.hint_layers[0](x)

        x = self.block2(x)
        hint2 = self.hint_layers[1](x)
        x = self.block3(x)
        x = self.block4(x)

        hint3 = self.hint_layers[2](x)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.output_layer(x)

        if self.return_output:
            return x
        else:
            return hint1, hint2, hint3 

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



class AllCNN(nn.Module):
    def __init__(self, scale, nb_class=10):
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
       
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.output_layer(x)

        return x 

