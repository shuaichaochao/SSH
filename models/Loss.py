import torch.nn as nn
import torch


def Binarry_Cross_Entropy(orginal_hash, hash_center):
    return -torch.sum(hash_center * torch.log(orginal_hash) + (1 - hash_center) * torch.log(1 - orginal_hash)) / len(
        hash_center)

class Loss(nn.Module):
    def __init__(self, code_length, gamma):
        super(Loss, self).__init__()
        self.code_length = code_length
        self.gamma = gamma
       

    def forward(self, orginal_hash, hash_center):

        orginal_Hash_center_loss = torch.mean((0.5*(orginal_hash + 1)-0.5*(hash_center + 1))**2)
        loss = orginal_Hash_center_loss
        return loss

  


