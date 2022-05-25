import torch
import torch.nn as nn
import torch.nn.functional as F



class SoftmaxFocalLoss(nn.Module):
    def __init__(self,gamma=2.0,weight=None,reduction="mean"):
        super().__init__()
        self.weight=weight
        self.gamma=gamma
        assert reduction in ["sum","mean","none"]
        self.reduction=reduction

    def forward(self,input,target):
        assert self.weight is None or isinstance(self.weight, torch.Tensor)
        ce = F.cross_entropy(input, target,reduction="none").view(-1)
        pt=torch.exp(-ce)
        if self.weight!=None:
            target=target.view(-1)
            weights=self.weight[target]
        else:
            weights=torch.ones_like(target).view(-1)

        focal=weights*((1-pt)**self.gamma)
        if self.reduction=="mean":
            return (focal*ce).sum()/weights.sum()
        
        elif self.reduction=="sum":
            return (focal*ce).sum()
        
        else:
            return focal*ce


if __name__ == "__main__":
    torch.manual_seed(123)

    weights=torch.Tensor([0.75,0.25])
    fl=SoftmaxFocalLoss(gamma=2,weight=weights,reduction="mean")

    #multiclass classfication
    pred=torch.rand(3,2)
    trg=torch.randint(0,2,(3,))
    a=fl.forward(pred,trg)
    print("1.multiclass classfication:\n")
    print(pred,"\n")
    print(trg,"\n")
    print("loss:",a,"\n")

    #multiclass segmentation
    pred=torch.rand(1,2,3,3)
    trg=torch.randint(0,2,(1,3,3))
    b=fl.forward(pred,trg)
    print("2.multiclass segmentation:\n")
    print(pred,"\n")
    print(trg,"\n")
    print("loss:",b,"\n")