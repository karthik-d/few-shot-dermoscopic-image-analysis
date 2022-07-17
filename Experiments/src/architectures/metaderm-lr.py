from .metaderm import MetaDerm 


class MetaDerm_LR(nn.Module):
    
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

    def forward(self, x, get_feat_vec=False):
        output = self.encoder(x)
        feature_vec = output.view(output.size(0), -1)
        # Pass through classifier
        if self.num_classes is not None:
            output = self.classifier()        
        # Conditional return
        if get_feat_vec:
            return feat_vec, output 
        else:
            return output  