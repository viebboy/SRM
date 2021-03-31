from torchvision.models import resnet as ResNetUtils
import Layers
import torch
import torch.nn as nn
import LabelResNet
import ResNet
import PredResNet
import SparseRepResNet


def resnet18(pretrained):
    return ResNet.resnet18(pretrained=pretrained)

def resnet34(pretrained):
    return ResNet.resnet34(pretrained=pretrained)

def sparse_rep_resnet34(codeword_multiplier, sparsity_multiplier):
    return SparseRepResNet.sparse_rep_resnet34(codeword_multiplier,
                                                sparsity_multiplier)


####
def resnet34_global_local_label(codeword_multiplier, sparsity_multiplier):
    return LabelResNet.resnet34GlobalLocalLabel(codeword_multiplier,
                                                sparsity_multiplier)

####
def resnet18_global_local_pred(codeword_multiplier, out_dims):
    return PredResNet.resnet18GlobalLocalPred(codeword_multiplier,
                                                out_dims)



