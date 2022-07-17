import torch.nn as nn


def conv_block(in_channels, out_channels):
    
    """
    Returns a Conv Block of the config - Conv-BN-ReLU-Pool
    """

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.MaxPool2d(2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class MetaDerm(nn.Module):
    
    """
    Contructs an architecture as described in the reference paper (MetaDermDiagnosis-KMahajan)

    ```
    (6 layers of)
    32 filters of size 3 x 3,
    and is followed by a 2 x 2 max pooling layer, batch nor-
    malization, and ReLU activation.
    ```
    """
    
    def __init__(self, x_dim=3, hid_dim=32, z_dim=32):
        super(MetaDerm, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
