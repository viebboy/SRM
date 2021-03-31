from torchvision.models import resnet as ResNetUtils
import torch
import torch.nn as nn
import ResNet
import AllCNN



def pred_allcnn(nb_codewords, scale, nb_class, return_output=False):
    return AllCNN.PredAllCNN(nb_codewords,
                             scale,
                             nb_class, 
                             return_output)

   
def hint_pred_allcnn(hint_dims, scale, nb_class, return_output=False):
    return AllCNN.HintPredAllCNN(hint_dims, scale, nb_class, return_output=return_output)

def allcnn(scale, nb_class):
    return AllCNN.AllCNN(scale, nb_class)

def resnext50(pretrained, nb_class):
    return ResNet.resnext50_32x4d(pretrained=pretrained,
                                   num_classes=nb_class)

def hint_label_resnext50(pretrained, nb_class):
    return ResNet.hint_label_resnext50_32x4d(pretrained=pretrained,
                                   num_classes=nb_class)


def label_resnext50(codeword_multiplier,
                     sparsity_multiplier,
                     nb_class,
                     pretrained):
    return ResNet.label_resnext50_32x4d(codeword_multiplier,
                                         sparsity_multiplier,
                                         nb_class,
                                         pretrained)

def sparse_rep_resnext50(codeword_multiplier,
                     sparsity_multiplier,
                     nb_class,
                     pretrained):
    return ResNet.sparse_rep_resnext50_32x4d(codeword_multiplier,
                                         sparsity_multiplier,
                                         nb_class,
                                         pretrained)

    
def resnet18(pretrained, nb_class):
    return ResNet.resnet18(pretrained=pretrained,
                           num_classes=nb_class)


   
def pred_resnet18(nb_codewords, nb_class, return_output=False):
    return ResNet.pred_resnet18(nb_codewords, nb_class, return_output)

def hint_pred_resnet18(hint_dims, nb_class, return_output=False):
    return ResNet.hint_pred_resnet18(hint_dims, nb_class, return_output)





