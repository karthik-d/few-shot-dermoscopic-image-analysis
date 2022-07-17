import torch.nn as nn

from .metaderm import MetaDerm 

# TODO: Transform into a common embedder, with backbone selection

class MetaDerm_LR(MetaDerm):
    
    """
    Contructs an architecture as described in the reference paper (MetaDermDiagnosis-KMahajan)

    ```
    (6 layers of)
    32 filters of size 3 x 3,
    and is followed by a 2 x 2 max pooling layer, batch nor-
    malization, and ReLU activation.
    ```
    """
    
    def __init__(self, x_dim=3, hid_dim=32, z_dim=32, num_classes=None):
        
        super(MetaDerm_LR, self).__init__()
        
        # Setup a linear transformation for classification
        self.num_classes = num_classes
        if self.num_classes is not None:
            self.classifier = nn.Linear(32, self.num_classes)

    
    def forward(self, x):
        
        print(x.shape)
        output = self.encoder(x).view(output.size(0), -1)
        print(output.shape)
        feature_vec = output
        
        # Pass through classifier
        if self.num_classes is not None:
            output = self.classifier(output)   
        else:
            output = output.view(output.size(0), -1)    
        
        return output