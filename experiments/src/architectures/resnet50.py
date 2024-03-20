from torchvision.models import resnet50

import torch.nn as nn


class ResNet50(nn.Module):
    
    """
    """
    
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        
        super(ResNet50, self).__init__()
        
        # load weights without final layer
        pretrain = resnet50(pretrained=True).state_dict()
        del pretrain["fc.weight"], pretrain["fc.bias"]

        self.encoder = resnet50()
        self.encoder.load_state_dict(state_dict=pretrain, strict=False)

        self.out_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()


    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
