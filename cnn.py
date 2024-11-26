from torch import nn

class CNNModel(nn.modules):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv3d
