import numpy as np
import torch
from torch import nn

# %%
# https://stackoverflow.com/questions/42883547/intuitive-understanding-of-1d-2d-and-3d-convolutions-in-convolutional-neural-n

class scTest(nn.Module):
    def __init__(self, **kwargs):
        super(scTest,self).__init__()

        self.dev = kwargs.get('device', torch.device('cpu'))
        #self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.seq_len = kwargs.get('seq_len', 1344)
        self.init_dim = kwargs.get('init_dim', 288)
        self.bottleneck_size = kwargs.get('bottleneck_size', 32)
        self.cell_num = kwargs.get('cell_num', 10000)
        self.batch_m = kwargs.get('batch_correction', None)

        ###########################################
        self.feature_layers = nn.ModuleList()
        ##1 CNN block
        self.feature_layers += [nn.Conv1d(in_channels=4, out_channels=self.init_dim, kernel_size=17, padding="same"),
                                nn.BatchNorm1d(self.init_dim,momentum=0.9), nn.GELU(), nn.MaxPool1d(kernel_size=3)]

        ##2 CNN tower
        in_dim = self.init_dim
        out_dim = self.init_dim
        for i in range(6):
            self.feature_layers += [nn.Conv1d(in_channels=int(round(in_dim)), out_channels=int(round(out_dim)),kernel_size=5,padding="same"),
                                    nn.BatchNorm1d(int(round(out_dim)), momentum=0.9), nn.GELU(), nn.MaxPool1d(2)]
            in_dim = out_dim
            out_dim *= 1.122

        ##3 CNN block
        self.feature_layers += [nn.Conv1d(in_channels=int(round(in_dim)), out_channels=256, kernel_size=1, padding="same"),
                                nn.BatchNorm1d(256,momentum=0.9), nn.GELU(), nn.Flatten()]
        ##4 Dense block
        feature_len = 256 * 7
        self.feature_layers += [nn.Linear(feature_len, self.bottleneck_size),
                                nn.BatchNorm1d(self.bottleneck_size, momentum=0.9),
                                nn.Dropout(p=0.2),
                                nn.GELU()]

        ##5 Final
        self.feature_layers += [nn.Linear(self.bottleneck_size, self.cell_num), nn.Sigmoid()]

        ###### Batch correction
        if self.batch_m:
            self.batch_info = torch.from_numpy(self.batch_m.values.transpose())  # batch matrix
            self.reg_layer1 = nn.Linear(self.bottleneck_size, self.cell_num)
            self.reg_layer2 = nn.Linear(self.bottleneck_size, self.cell_num)

    def forward(self, x):
        if self.batch_m:
            model = nn.Sequential(*self.feature_layers)
            x = model(x)
            y1 = self.reg_layer1(x)
            y2 = self.reg_layer2(x)
            y2 = torch.linalg.matmul(y2, self.batch_info)
            y = torch.add(y1,y2)
            y = torch.sigmoid(y)
        else:
            model = nn.Sequential(*self.feature_layers)
            y = model(x)
        return y
#####
seq_len = 1344
model = scTest()
x = torch.randn(64,seq_len,4) ## batch size, feature_len, embedding_size
x = x.permute(0,2,1)
out = model(x)






