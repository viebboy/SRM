import math
import Layers
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out



class GlobalLocalLabelDenseNet(nn.Module):
    def __init__(self, 
                codeword_multiplier,
                sparsity_multiplier,
                centers,
                intercept,
                block, 
                nblocks, 
                growth_rate=12, 
                reduction=0.5, 
                nb_class=10):
        
        super(GlobalLocalLabelDenseNet, self).__init__()
        
        in_dims = []


        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        in_dims.append(num_planes)

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        in_dims.append(num_planes)

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        in_dims.append(num_planes)
        
        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, nb_class)


        if centers is None:
            centers = [None,]*3
        if intercept is None:
            intercept = [None,]*3

        self.sparse_layers = nn.ModuleList()
        sparsity_multiplier *= codeword_multiplier
        self.sparse_layers.append(Layers.GlobalLocalLabel(in_dims[0],
                                                 in_dims[0]*codeword_multiplier,
                                                 int(in_dims[0]*sparsity_multiplier),
                                                 centers=centers[0],
                                                 intercept=intercept[0]))

        self.sparse_layers.append(Layers.GlobalLocalLabel(in_dims[1],
                                                 in_dims[1]*codeword_multiplier,
                                                 int(in_dims[1]*sparsity_multiplier),
                                                 centers=centers[1],
                                                 intercept=intercept[1]))

        self.sparse_layers.append(Layers.GlobalLocalLabel(in_dims[2],
                                                 in_dims[2]*codeword_multiplier,
                                                 int(in_dims[2]*sparsity_multiplier),
                                                 centers=centers[2],
                                                 intercept=intercept[2]))


        self.initialize()

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out1 = self.trans1(self.dense1(out)) # 16x16 
        ghist1, lhist1 = self.sparse_layers[0](out1)

        out2 = self.trans2(self.dense2(out1)) # 8x8 
        ghist2, lhist2 = self.sparse_layers[1](out2)
        
        out = self.trans3(self.dense3(out2)) # 4x4

        out3 = self.dense4(out) # 4x4
        ghist3, lhist3 = self.sparse_layers[2](out3)
        
 
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


class LocalLabelDenseNet(nn.Module):
    def __init__(self, 
                codeword_multiplier,
                sparsity_multiplier,
                centers,
                intercept,
                block, 
                nblocks, 
                growth_rate=12, 
                reduction=0.5, 
                nb_class=10):
        
        super(LocalLabelDenseNet, self).__init__()
        
        in_dims = []

        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        in_dims.append(num_planes)

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        in_dims.append(num_planes)

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        in_dims.append(num_planes)
        
        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, nb_class)


        # nbof layers

        if centers is None:
            centers = [None,]*3
        if intercept is None:
            intercept = [None,]*3

        self.sparse_layers = nn.ModuleList()
        sparsity_multiplier *= codeword_multiplier
        self.nb_codewords = []
        self.sparse_layers.append(Layers.LocalLabel(in_dims[0],
                                                 in_dims[0]*codeword_multiplier,
                                                 int(in_dims[0]*sparsity_multiplier),
                                                 centers=centers[0],
                                                 intercept=intercept[0]))

        self.sparse_layers.append(Layers.LocalLabel(in_dims[1],
                                                 in_dims[1]*codeword_multiplier,
                                                 int(in_dims[1]*sparsity_multiplier),
                                                 centers=centers[1],
                                                 intercept=intercept[1]))

        self.sparse_layers.append(Layers.LocalLabel(in_dims[2],
                                                 in_dims[2]*codeword_multiplier,
                                                 int(in_dims[2]*sparsity_multiplier),
                                                 centers=centers[2],
                                                 intercept=intercept[2]))

        self.nb_codewords.append(int(in_dims[0]*codeword_multiplier))
        self.nb_codewords.append(int(in_dims[1]*codeword_multiplier))
        self.nb_codewords.append(int(in_dims[2]*codeword_multiplier))
        
        self.initialize()

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out1 = self.trans1(self.dense1(out)) # 16x16 
        hist1 = self.sparse_layers[0](out1)

        out2 = self.trans2(self.dense2(out1)) # 8x8 
        hist2 = self.sparse_layers[1](out2)

        out = self.trans3(self.dense3(out2)) # 4x4

        out3 = self.dense4(out) # 4x4
        hist3 = self.sparse_layers[2](out3)

        
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


class GlobalLabelDenseNet(nn.Module):
    def __init__(self, 
                codeword_multiplier,
                sparsity_multiplier,
                centers,
                intercept,
                block, 
                nblocks, 
                growth_rate=12, 
                reduction=0.5, 
                nb_class=10):
        
        super(GlobalLabelDenseNet, self).__init__()
        
        in_dims = []

        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        in_dims.append(num_planes)

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        in_dims.append(num_planes)

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        in_dims.append(num_planes)
        
        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, nb_class)


        # nbof layers

        if centers is None:
            centers = [None,]*3
        if intercept is None:
            intercept = [None,]*3

        self.sparse_layers = nn.ModuleList()
        sparsity_multiplier *= codeword_multiplier
        self.sparse_layers.append(Layers.GlobalLabel(in_dims[0],
                                                 in_dims[0]*codeword_multiplier,
                                                 int(in_dims[0]*sparsity_multiplier),
                                                 centers=centers[0],
                                                 intercept=intercept[0]))

        self.sparse_layers.append(Layers.GlobalLabel(in_dims[1],
                                                 in_dims[1]*codeword_multiplier,
                                                 int(in_dims[1]*sparsity_multiplier),
                                                 centers=centers[1],
                                                 intercept=intercept[1]))

        self.sparse_layers.append(Layers.GlobalLabel(in_dims[2],
                                                 in_dims[2]*codeword_multiplier,
                                                 int(in_dims[2]*sparsity_multiplier),
                                                 centers=centers[2],
                                                 intercept=intercept[2]))


        self.initialize()

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out1 = self.trans1(self.dense1(out)) # 16x16 
        hist1 = self.sparse_layers[0](out1)

        out2 = self.trans2(self.dense2(out1)) # 8x8 
        hist2 = self.sparse_layers[1](out2)

        out = self.trans3(self.dense3(out2)) # 4x4

        out3 = self.dense4(out) # 4x4
        hist3 = self.sparse_layers[2](out3)

        
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



class SparseRepDenseNet(nn.Module):
    def __init__(self, 
                codeword_multiplier,
                sparsity_multiplier,
                centers,
                intercept,
                block, 
                nblocks, 
                growth_rate=12, 
                reduction=0.5, 
                nb_class=10):
        
        super(SparseRepDenseNet, self).__init__()
        
        in_dims = []

        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        in_dims.append(num_planes)

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        in_dims.append(num_planes)

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        in_dims.append(num_planes)
        
        #print(in_dims)

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, nb_class)


        # nbof layers

        if centers is None:
            centers = [None,]*3
        if intercept is None:
            intercept = [None,]*3

        self.sparse_layers = nn.ModuleList()
        sparsity_multiplier = sparsity_multiplier*codeword_multiplier
        
        self.sparse_layers.append(Layers.SparseRep(in_dims[0],
                                                 in_dims[0]*codeword_multiplier,
                                                 int(in_dims[0]*sparsity_multiplier),
                                                 centers=centers[0],
                                                 intercept=intercept[0]))

        self.sparse_layers.append(Layers.SparseRep(in_dims[1],
                                                 in_dims[1]*codeword_multiplier,
                                                 int(in_dims[1]*sparsity_multiplier),
                                                 centers=centers[1],
                                                 intercept=intercept[1]))

        self.sparse_layers.append(Layers.SparseRep(in_dims[2],
                                                 in_dims[2]*codeword_multiplier,
                                                 int(in_dims[2]*sparsity_multiplier),
                                                 centers=centers[2],
                                                 intercept=intercept[2]))


        self.initialize()


    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.no_grad():
            out = self.conv1(x)
            out1 = self.trans1(self.dense1(out)) # 16x16 
            out2 = self.trans2(self.dense2(out1)) # 8x8 
            out = self.trans3(self.dense3(out2)) # 4x4
            out3 = self.dense4(out) # 4x4
        

        bof1 = self.sparse_layers[0](out1) 
        bof2 = self.sparse_layers[1](out2) 
        bof3 = self.sparse_layers[2](out3)

        return (out1, out2, out3), (bof1, bof2, bof3)

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


class HintDenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, nb_class=10):
        super(HintDenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, nb_class)

        self.initialize()


    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out1 = self.trans1(self.dense1(out))
        
        out2 = self.trans2(self.dense2(out1))
        out3 = self.trans3(self.dense3(out2))
        out4 = self.dense4(out3)
        
        return out1, out2, out4

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


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, nb_class=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, nb_class)

        self.initialize()


    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        #print(out.size())
        out = self.trans1(self.dense1(out))
        #print(out.size())
        out = self.trans2(self.dense2(out))
        #print(out.size())
        out = self.trans3(self.dense3(out))
        #print(out.size())
        out = self.dense4(out)
        #print(out.size())
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

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

def GlobalLabel(codeword_multiplier, 
                sparsity_multiplier, 
                centers, 
                intercept, 
                nb_class):
    return GlobalLabelDenseNet(codeword_multiplier,
                               sparsity_multiplier,
                               centers,
                               intercept,
                               Bottleneck, 
                               [6,12,24,16], 
                               growth_rate=32, 
                               nb_class=nb_class)


def LocalLabel(codeword_multiplier, 
               sparsity_multiplier,
               centers, 
               intercept, 
               nb_class):
    return LocalLabelDenseNet(codeword_multiplier,
                              sparsity_multiplier,
                              centers,
                              intercept,
                              Bottleneck, 
                              [6,12,24,16], 
                              growth_rate=32, 
                              nb_class=nb_class)


    
def GlobalLocalLabel(codeword_multiplier, 
                     sparsity_multiplier,
                     centers, 
                     intercept, 
                     nb_class):
    return GlobalLocalLabelDenseNet(codeword_multiplier,
                                    sparsity_multiplier,
                                    centers,
                                    intercept,
                                    Bottleneck, 
                                    [6,12,24,16], 
                                    growth_rate=32, 
                                    nb_class=nb_class)

  
def SparseRepDenseNet121(codeword_multiplier, 
                     sparsity_multiplier,
                     centers, 
                     intercept, 
                     nb_class):
    return SparseRepDenseNet(codeword_multiplier,
                            sparsity_multiplier,
                            centers,
                            intercept,
                            Bottleneck, 
                            [6,12,24,16], 
                            growth_rate=32, 
                            nb_class=nb_class)

   
def DenseNet121(nb_class):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32, nb_class=nb_class)


def HintDenseNet121(nb_class):
    return HintDenseNet(Bottleneck, [6,12,24,16], growth_rate=32, nb_class=nb_class)


