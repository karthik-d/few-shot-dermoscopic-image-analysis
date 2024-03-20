import torch.nn as nn

from .resnet18 import ResNet18

# TODO: Transform into a common embedder, with backbone selection

class ResNet18_LR(ResNet18):
    
    """
    ```
    """
    
    def __init__(self, x_dim=3, hid_dim=32, z_dim=32, num_classes=None):
        
        super(ResNet18_LR, self).__init__()
        
        # Setup a linear transformation for classification
        self.num_classes = num_classes
        if self.num_classes is not None:
            self.classifier = nn.Linear(self.out_features, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
    def forward(self, x):
        
        output = self.encoder(x)
        
        # Pass through classifier
        if self.num_classes is not None:
            feat_vec = output.view(output.size(0), -1)
            output = self.classifier(feat_vec)  
        else:
            output = output.view(output.size(0), -1)
        
        return output