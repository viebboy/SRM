import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        # N x H x W x nb_codeword
        similarity = self.hyperbolic(x)
      
        # global_hist: N x nb_codeword
        global_pred = torch.mean(similarity, dim=(1, 2))
        # local_hist: N x nb_codeword x H x W
        
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
        # convert x from N x D x H x W to N x W x H x D   
        x = x.transpose(1, -1)
        dist = torch.sigmoid(F.linear(x, self.centers.transpose(0, 1)) + self.intercept)
        dist = dist.transpose(1, 2)
        return dist

    def forward(self, x):
        
        # N x H x W x nb_codeword
        similarity = self.hyperbolic(x)
      
        coef, _ = torch.topk(similarity,
                             self.sparsity,
                             dim=-1)

        min_coef = coef[:, :, :, -1].unsqueeze(-1)

        similarity = similarity * (similarity >= min_coef).float()

        # global_hist: N x nb_codeword
        global_label = torch.mean(similarity, dim=(1, 2))
        # local_hist: N x nb_codeword x H x W
        
        local_label = similarity.transpose(2, 3).transpose(1, 2)
        local_label = torch.argmax(local_label, dim=1)
       
        return global_label, local_label 



class LocalPred(nn.Module):
    def __init__(self,
                in_dim,
                nb_codeword,
                centers=None,
                intercept=None):

        super(LocalPred, self).__init__()

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
        similarity = similarity.transpose(2, 3).transpose(1, 2)
        return similarity


class GlobalPred(nn.Module):
    def __init__(self,
                in_dim,
                nb_codeword,
                centers=None,
                intercept=None):

        super(GlobalPred, self).__init__()

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
        return torch.mean(similarity, dim=(1, 2))


class LocalLabel(nn.Module):
    def __init__(self,
                in_dim,
                nb_codeword,
                sparsity,
                centers=None,
                intercept=None):

        super(LocalLabel, self).__init__()

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
        similarity = similarity.transpose(2, 3).transpose(1, 2)
        similarity = torch.argmax(similarity, dim=1)
        return similarity 


class GlobalLabel(nn.Module):
    def __init__(self,
                in_dim,
                nb_codeword,
                sparsity,
                centers=None,
                intercept=None):

        super(GlobalLabel, self).__init__()

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
        # coef: N x H x W x sparsity
        coef, _ = torch.topk(similarity, 
                             self.sparsity,
                             dim=-1)
        # min coef: N x H x W x 1
        min_coef = coef[:, :, :, -1].unsqueeze(-1)

        similarity = similarity * (similarity >= min_coef).float()
       
        return torch.mean(similarity, dim=(1, 2))



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

        # use the topk similarity & centers to reconstruct features 
        # dist: N x H x W x nb_codeword 
        # centers: D x nb_codeword
        #
        coef, indices = torch.topk(similarity,
                                   self.sparsity,
                                   dim=-1)


        # coef: N x H x W x sparsity 
        # now find the binary mask that select topk values
        min_coef = coef[:, :, :, -1].unsqueeze(-1)
        
        similarity = similarity * (similarity >= min_coef).float()

        x = F.linear(similarity, self.centers)

        # now swap the axis
        x = x.unsqueeze(1)
        x = x.transpose(1, -1)
        x = x.view(x.size(0),
                   x.size(1),
                   x.size(2),
                   x.size(3))

        
        return x

