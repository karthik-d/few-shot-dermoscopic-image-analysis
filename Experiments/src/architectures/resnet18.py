from torchvision.models import resnet18, ResNet18_Weights

import torch.nn as nn


class ResNet18(nn.Module):
    
    """
    """
    
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        
        super(ResNet18, self).__init__()
        
        # load weights without final layer
        pretrain = resnet18(pretrained=True).state_dict()
        del pretrained["fc.weight"], pretrained["fc.bias"]

        self.encoder = resnet18(weights=weights, progress=False).eval()
        self.encoder.load_state_dict(state_dict=pretrained, strict=False)
        

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
