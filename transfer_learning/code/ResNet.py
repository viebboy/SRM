from torchvision.models import resnet as ResNetUtils
import Layers
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)


        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class HintPredResNet(nn.Module):

    def __init__(self,
                 hint_dims,
                 nb_class,
                 return_output,
                 block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(HintPredResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        self.return_output = return_output

        in_dims = []
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        in_dims.append(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])

        in_dims.append(self.inplanes)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        in_dims.append(self.inplanes)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        in_dims.append(self.inplanes)
        in_dims.append(self.inplanes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_layer = nn.Linear(512 * block.expansion, nb_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


        self.hint_layers = nn.ModuleList()
        for idx, input_dim in enumerate(in_dims):
            self.hint_layers.append(nn.Conv2d(in_channels=input_dim,
                                              out_channels=hint_dims[idx],
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))




    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        if self.return_output:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)


            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.output_layer(x)
            return x
        
        out1 = self.conv1(x)
        x = self.bn1(out1)
        x = self.relu(x)
        out2 = self.maxpool(x)

        out3 = self.layer1(out2)
        out4 = self.layer2(out3)
        out5 = self.layer3(out4)
        out6 = self.layer4(out5)

        out7 = self.avgpool(out6)

        hiddens = [out2, out4, out5, out6, out7]
        local_pred = []

        for idx, layer in enumerate(self.hint_layers):
            hout = layer(hiddens[idx])
            local_pred.append(hout)

        return local_pred


    def forward(self, x):
        return self._forward_impl(x)


class PredResNet(nn.Module):

    def __init__(self,
                 nb_codewords,
                 nb_class,
                 return_output,
                 block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(PredResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.return_output = return_output
        in_dims = []
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        in_dims.append(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])

        in_dims.append(self.inplanes)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        in_dims.append(self.inplanes)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        in_dims.append(self.inplanes)
        in_dims.append(self.inplanes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_layer = nn.Linear(512 * block.expansion, nb_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


        self.sparse_layers = nn.ModuleList()
        for idx, input_dim in enumerate(in_dims):
            self.sparse_layers.append(Layers.GlobalLocalPred(input_dim,
                                                       nb_codewords[idx]))




    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        if self.return_output:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)


            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.output_layer(x)
            return x

        out1 = self.conv1(x)
        x = self.bn1(out1)
        x = self.relu(x)
        out2 = self.maxpool(x)

        out3 = self.layer1(out2)
        out4 = self.layer2(out3)
        out5 = self.layer3(out4)
        out6 = self.layer4(out5)

        out7 = self.avgpool(out6)

        hiddens = [out2, out4, out5, out6, out7]
        global_pred = []
        local_pred = []

        for idx, layer in enumerate(self.sparse_layers):
            g_out, l_out = layer(hiddens[idx])
            global_pred.append(g_out)
            local_pred.append(l_out)

        return global_pred, local_pred


    def forward(self, x):
        return self._forward_impl(x)



class LabelResNet(nn.Module):

    def __init__(self,
                 codeword_multiplier,
                 sparsity_multiplier,
                 nb_class,
                 block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(LabelResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer


        in_dims = []
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        in_dims.append(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0])

        

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])

        in_dims.append(self.inplanes)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        in_dims.append(self.inplanes)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        in_dims.append(self.inplanes)
        in_dims.append(self.inplanes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, nb_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


        sparsity_multiplier *= codeword_multiplier
        self.sparse_layers = nn.ModuleList()
        self.nb_codewords = []
        for input_dim in in_dims:
            self.sparse_layers.append(Layers.GlobalLocalLabel(input_dim,
                                                       input_dim*codeword_multiplier,
                                                       input_dim*sparsity_multiplier))

            nb_codeword = int(input_dim*codeword_multiplier)
            self.nb_codewords.append(nb_codeword)



    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        out1 = self.conv1(x)
        x = self.bn1(out1)
        x = self.relu(x)
        out2 = self.maxpool(x)

        out3 = self.layer1(out2)
        out4 = self.layer2(out3)
        out5 = self.layer3(out4)
        out6 = self.layer4(out5)

        out7 = self.avgpool(out6)

        hiddens = [out2, out4, out5, out6, out7]
        global_labels = []
        local_labels = []

        for idx, layer in enumerate(self.sparse_layers):
            g_out, l_out = layer(hiddens[idx])
            global_labels.append(g_out)
            local_labels.append(l_out)

        return global_labels, local_labels


    def forward(self, x):
        return self._forward_impl(x)


class SparseRepResNet(nn.Module):

    def __init__(self,
                 codeword_multiplier,
                 sparsity_multiplier,
                 nb_class,
                 block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(SparseRepResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer


        in_dims = []
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        in_dims.append(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])

        in_dims.append(self.inplanes)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        in_dims.append(self.inplanes)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        in_dims.append(self.inplanes)
        in_dims.append(self.inplanes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, nb_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


        sparsity_multiplier *= codeword_multiplier
        self.sparse_layers = nn.ModuleList()
        self.nb_codewords = []
        for input_dim in in_dims:
            self.sparse_layers.append(Layers.SparseRep(input_dim,
                                                       input_dim*codeword_multiplier,
                                                       input_dim*sparsity_multiplier))

            nb_codeword = int(input_dim*codeword_multiplier)
            self.nb_codewords.append(nb_codeword)



    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        out1 = self.conv1(x)
        x = self.bn1(out1)
        x = self.relu(x)
        out2 = self.maxpool(x)
        
        out3 = self.layer1(out2)
        out4 = self.layer2(out3)
        out5 = self.layer3(out4)
        out6 = self.layer4(out5)

        out7 = self.avgpool(out6)

        hiddens = [out2, out4, out5, out6, out7]
        reconstructed = []

        for idx, layer in enumerate(self.sparse_layers):
            recon = layer(hiddens[idx])
            reconstructed.append(recon)

        return hiddens, reconstructed 


    def forward(self, x):
        return self._forward_impl(x)


class HintLabelResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(HintLabelResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.hint_dims = []
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.hint_dims.append(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0])
        

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])

        self.hint_dims.append(self.inplanes)
        
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        self.hint_dims.append(self.inplanes)
        
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        self.hint_dims.append(self.inplanes)
        self.hint_dims.append(self.inplanes)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        out2 = self.maxpool(x)

        out3 = self.layer1(out2)
        out4 = self.layer2(out3)
        out5 = self.layer3(out4)
        out6 = self.layer4(out5)

        out7 = self.avgpool(out6)

        return out2, out4, out5, out6, out7

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, num_classes, **kwargs):
    model = ResNet(block, layers, num_classes=num_classes, **kwargs)
    if pretrained:
        state_dict = ResNetUtils.load_state_dict_from_url(ResNetUtils.model_urls[arch],
                                              progress=progress)
        cur_state_dict = model.state_dict()
        count = 0
        for layer in cur_state_dict.keys():
            if 'fc' not in layer and layer in state_dict.keys():
                cur_state_dict[layer] = state_dict[layer]
                count += 1
            else:
                print('skipping layer: %s' % str(layer))

        print('load weights from %s layers' % str(count))
        model.load_state_dict(cur_state_dict)
    return model


def _hint_label_resnet(arch, block, layers, pretrained, progress, num_classes, **kwargs):
    model = HintLabelResNet(block, layers, num_classes=num_classes, **kwargs)
    if pretrained:
        state_dict = ResNetUtils.load_state_dict_from_url(ResNetUtils.model_urls[arch],
                                              progress=progress)
        cur_state_dict = model.state_dict()
        count = 0
        for layer in cur_state_dict.keys():
            if 'fc' not in layer and layer in state_dict.keys():
                cur_state_dict[layer] = state_dict[layer]
                count += 1
            else:
                print('skipping layer: %s' % str(layer))

        print('load weights from %s layers' % str(count))
        model.load_state_dict(cur_state_dict)
    return model


def _label_resnet(codeword_multiplier, sparsity_multiplier,  nb_class,
        arch, block, layers, pretrained, progress, **kwargs):
    model = LabelResNet(codeword_multiplier, sparsity_multiplier,  nb_class,
            block, layers, **kwargs)
    if pretrained:
        state_dict = ResNetUtils.load_state_dict_from_url(ResNetUtils.model_urls[arch],
                                              progress=progress)
        cur_state_dict = model.state_dict()
        count = 0
        for layer in cur_state_dict.keys():
            if 'fc' not in layer and layer in state_dict.keys():
                cur_state_dict[layer] = state_dict[layer]
                count += 1
            else:
                print('skipping layer: %s' % str(layer))

        print('load weights from %s layers' % str(count))
        model.load_state_dict(cur_state_dict)
    
    return model

def _sparse_rep_resnet(codeword_multiplier, sparsity_multiplier,  nb_class,
        arch, block, layers, pretrained, progress, **kwargs):
    model = SparseRepResNet(codeword_multiplier, sparsity_multiplier, nb_class,
            block, layers, **kwargs)
    if pretrained:
        state_dict = ResNetUtils.load_state_dict_from_url(ResNetUtils.model_urls[arch],
                                              progress=progress)
        cur_state_dict = model.state_dict()
        count = 0
        for layer in cur_state_dict.keys():
            if 'fc' not in layer and layer in state_dict.keys():
                cur_state_dict[layer] = state_dict[layer]
                count += 1
            else:
                print('skipping layer: %s' % str(layer))

        print('load weights from %s layers' % str(count))
        model.load_state_dict(cur_state_dict)
    
    return model

def _pred_resnet(nb_codewords, nb_class, return_output,
        arch, block, layers, **kwargs):
    model = PredResNet(nb_codewords, nb_class, return_output,
            block, layers, **kwargs)
    
    return model

def _hint_pred_resnet(hint_dims, nb_class, return_output,
        arch, block, layers, **kwargs):
    model = HintPredResNet(hint_dims, nb_class, return_output,
            block, layers, **kwargs)
    
    return model




def resnet18(num_classes, pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', ResNetUtils.BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   num_classes=num_classes, **kwargs)

    
    
def pred_resnet18(nb_codewords,
                  nb_class, 
                  return_output, **kwargs):
    
    
    return _pred_resnet(nb_codewords,  
            nb_class, return_output, 'resnet18', 
            ResNetUtils.BasicBlock, [2, 2, 2, 2], **kwargs)

def hint_pred_resnet18(hint_dims, nb_class, return_output, **kwargs):
    return _hint_pred_resnet(hint_dims, nb_class, 
            return_output, 'resnet18', ResNetUtils.BasicBlock, [2, 2, 2, 2], **kwargs)

    
def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', ResNetUtils.BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(num_classes, pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', ResNetUtils.Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   num_classes, **kwargs)

def resnext101_32x8d(num_classes, pretrained=True, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', ResNetUtils.Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, num_classes, **kwargs)

def resnext50_32x4d(num_classes, pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', ResNetUtils.Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, num_classes, **kwargs)
    
def hint_label_resnext50_32x4d(num_classes, pretrained=False, progress=True, **kwargs):
    
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _hint_label_resnet('resnext50_32x4d', ResNetUtils.Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, num_classes, **kwargs)
 
    

def label_resnext101_32x8d(codeword_multiplier,
                          sparsity_multiplier,
                          nb_class,
                          pretrained=True, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _label_resnet(codeword_multiplier,
                        sparsity_multiplier,
                        nb_class,
                        'resnext101_32x8d', 
                        ResNetUtils.Bottleneck, 
                        [3, 4, 23, 3],
                        pretrained, progress, **kwargs)

def label_resnext50_32x4d(codeword_multiplier,
                          sparsity_multiplier,
                          nb_class,
                          pretrained=True, progress=True, **kwargs):
    
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _label_resnet(codeword_multiplier,
                        sparsity_multiplier,
                        nb_class,
                        'resnext50_32x4d', 
                        ResNetUtils.Bottleneck, 
                        [3, 4, 6, 3],
                        pretrained, progress, **kwargs)

def sparse_rep_resnext50_32x4d(codeword_multiplier,
                          sparsity_multiplier,
                          nb_class,
                          pretrained=True, progress=True, **kwargs):
    
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _sparse_rep_resnet(codeword_multiplier,
                        sparsity_multiplier,
                        nb_class,
                        'resnext50_32x4d', 
                        ResNetUtils.Bottleneck, 
                        [3, 4, 6, 3],
                        pretrained, progress, **kwargs)


