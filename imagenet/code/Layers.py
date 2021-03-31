import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomParallel(nn.DataParallel):


    def state_dict(self,):
        weights = super(CustomParallel, self).state_dict()
        new_weights = {}
        for layer in weights.keys():
            new_weights[layer[7:]] = weights[layer]

        return new_weights

    def load_state_dict(self, weights):
        new_weights = {}
        for layer in weights.keys():
            new_weights['module.' + layer] = weights[layer]

        super(CustomParallel, self).load_state_dict(new_weights)



class GlobalLocalPred(nn.Module):
    def __init__(self,
            in_dim,
            nb_codeword,
            centers=None,
            intercept=None):

        super(GlobalLocalPred, self).__init__()

        nb_codeword = int(nb_codeword)

        self.nb_codeword = nb_codeword

        if centers is not None:
            centers = centers.clone().detach()
            self.centers = nn.Parameter(data=centers,
                                requires_grad=True)

        else:
            self.centers = nn.Parameter(data=torch.Tensor(in_dim, nb_codeword),
                                requires_grad=True)

            nn.init.kaiming_uniform_(self.centers)

        if intercept is None:
            self.intercept = nn.Parameter(data=torch.Tensor(1,),
                            requires_grad=True)
            nn.init.uniform_(self.intercept, -1.0, 1.0)
        else:
            intercept = intercept.clone().detach()
            self.intercept = nn.Parameter(data=intercept,
                            requires_grad=True)
       
                
    def hyperbolic(self, x):
        x = x.transpose(1, -1)
        dist = torch.sigmoid(F.linear(x, self.centers.transpose(0, 1)) + self.intercept)
        dist = dist.transpose(1, 2)

        return dist

    def forward(self, x):
        
        similarity = self.hyperbolic(x)
      
        global_pred = torch.mean(similarity, dim=(1, 2))
        
        local_pred = similarity.transpose(2, 3).transpose(1, 2)
        
        return global_pred, local_pred 



class GlobalLocalLabel(nn.Module):
    def __init__(self,
            in_dim,
            nb_codeword,
            sparsity,
            centers=None,
            intercept=None):

        super(GlobalLocalLabel, self).__init__()

        nb_codeword = int(nb_codeword)
        sparsity = int(sparsity)
        if sparsity == 0:
            sparsity = 1

        self.sparsity = sparsity
        self.nb_codeword = nb_codeword

        if centers is not None:
            centers = centers.clone().detach()
            self.centers = nn.Parameter(data=centers,
                                requires_grad=True)

        else:
            self.centers = nn.Parameter(data=torch.Tensor(in_dim, nb_codeword),
                                requires_grad=True)

            nn.init.kaiming_uniform_(self.centers)

        if intercept is None:
            self.intercept = nn.Parameter(data=torch.Tensor(1,),
                            requires_grad=True)
            nn.init.uniform_(self.intercept, -1.0, 1.0)
        else:
            intercept = intercept.clone().detach()
            self.intercept = nn.Parameter(data=intercept,
                            requires_grad=True)
   
        
    def hyperbolic(self, x):
        x = x.transpose(1, -1)
        dist = torch.sigmoid(F.linear(x, self.centers.transpose(0, 1)) + self.intercept)
        dist = dist.transpose(1, 2)
        return dist

    def forward(self, x):
        
        similarity = self.hyperbolic(x)
      
        coef, _ = torch.topk(similarity,
                             self.sparsity,
                             dim=-1)

        min_coef = coef[:, :, :, -1].unsqueeze(-1)

        similarity = similarity * (similarity >= min_coef).float()

        global_label = torch.mean(similarity, dim=(1, 2))
        local_label = similarity.transpose(2, 3).transpose(1, 2)
        local_label = torch.argmax(local_label, dim=1)
       
        return global_label, local_label 




class SparseRep(nn.Module):
    def __init__(self,
                in_dim,
                nb_codeword,
                sparsity,
                centers=None,
                intercept=None):

        super(SparseRep, self).__init__()

        nb_codeword = int(nb_codeword)
        sparsity = int(sparsity)
        if sparsity == 0:
            sparsity = 1

        self.sparsity = sparsity 
        self.nb_codeword = nb_codeword

        if centers is not None:
            centers = centers.clone().detach()
            self.centers = nn.Parameter(data=centers,
                                requires_grad=True)

        else:
            self.centers = nn.Parameter(data=torch.Tensor(in_dim, nb_codeword),
                                requires_grad=True)

            nn.init.kaiming_uniform_(self.centers)

        if intercept is None:
            self.intercept = nn.Parameter(data=torch.Tensor(1,),
                                requires_grad=True)
       
            nn.init.uniform_(self.intercept, -1.0, 1.0)
        else:
            intercept = intercept.clone().detach()
            self.intercept = nn.Parameter(data=intercept,
                                requires_grad=True)

            

    def hyperbolic(self, x):
        x = x.transpose(1, -1)
        dist = torch.sigmoid(F.linear(x, self.centers.transpose(0, 1)) + self.intercept)
        dist = dist.transpose(1, 2)
        return dist

    def forward(self, x):
        similarity = self.hyperbolic(x)

        coef, indices = torch.topk(similarity,
                                   self.sparsity,
                                   dim=-1)
        min_coef = coef[:, :, :, -1].unsqueeze(-1)
        
        similarity = similarity * (similarity >= min_coef).float()

        x = F.linear(similarity, self.centers)

        x = x.unsqueeze(1)
        x = x.transpose(1, -1)
        x = x.view(x.size(0),
                   x.size(1),
                   x.size(2),
                   x.size(3))

        return x

