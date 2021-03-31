from torchvision.models import resnet as ResNetUtils
import Layers
import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class GlobalLocalPredResNet(nn.Module):

    def __init__(self,
                 codeword_multiplier,
                 out_dims,
                 block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(GlobalLocalPredResNet, self).__init__()
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

        self.layer1 = self._make_layer(block, 64, layers[0])

        in_dims.append(self.inplanes)

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



        self.sparse_layers = nn.ModuleList()

        for input_dim, output_dim in zip(in_dims, out_dims):
            self.sparse_layers.append(Layers.GlobalLocalPred(input_dim,
                                                       output_dim*codeword_multiplier))



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
        global_preds = []
        local_preds = []

        for idx, layer in enumerate(self.sparse_layers):
            g_out, l_out = layer(hiddens[idx])
            global_preds.append(g_out)
            local_preds.append(l_out)

        return global_preds, local_preds


    def forward(self, x):
        return self._forward_impl(x)



def _resnet_global_local_pred(codeword_multiplier, out_dims, 
        arch, block, layers, pretrained, progress, **kwargs):
    model = GlobalLocalPredResNet(codeword_multiplier, out_dims, 
            block, layers, **kwargs)
    if pretrained:
        state_dict = ResNetUtils.load_state_dict_from_url(ResNetUtils.model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18GlobalLocalPred(codeword_multiplier, out_dims,
        pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_global_local_pred(codeword_multiplier, out_dims, 
            'resnet18', ResNetUtils.BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)



