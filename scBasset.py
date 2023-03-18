import os.path
import time

import numpy as np
import torch
from torch import nn

class scBasset(nn.Module):
    def __init__(self, **kwargs):
        super(scBasset,self).__init__()

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
        self.seq_embedding = nn.Sequential(*self.feature_layers)  ## network for peak sequence embeding

        ##5 Final layer for output
        self.final_layer = nn.Sequential(nn.Linear(self.bottleneck_size, self.cell_num), nn.Sigmoid())

        ###### Batch correction
        if self.batch_m:
            self.batch_info = torch.from_numpy(self.batch_m.values.transpose())  # batch matrix
            self.reg_layer1 = nn.Linear(self.bottleneck_size, self.cell_num)
            self.reg_layer2 = nn.Linear(self.bottleneck_size, self.cell_num)

    def forward(self, x):
        if self.batch_m:
            s_emb = self.seq_embedding(x)
            y1 = self.reg_layer1(s_emb)
            y2 = self.reg_layer2(s_emb)
            y2 = torch.linalg.matmul(y2, self.batch_info)
            y = torch.add(y1,y2)
            y = torch.sigmoid(y)
        else:
            s_emb = self.seq_embedding(x)
            y = self.final_layer(s_emb)
        return y

    def save(self,path=None):
        if path:
            path = os.path.join(path,"trained_model.pt")
        torch.save(self.state_dict(), path)
        print("Saved PyTorch Model State to ",path)

    def train_model(self, train_loader, learning_rate, n_epochs, val_loader=False):
        """ Train NaiveCNN """
        ## Record loss
        epoch_hist = {'train_loss':[],'validation_loss':[]}

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_func = nn.CrossEntropyLoss()

        # Train
        for epoch in range(n_epochs):
            t0 = time.time()
            loss_value = 0
            self.train()
            ## batches
            for x_train,y_train in train_loader:
                x_train = x_train.to(self.dev,dtype=torch.float32)
                y_train = y_train.to(self.dev,dtype=torch.float32)

                # Compute prediction error
                y_train_hat = self.forward(x_train)
                loss = loss_func(y_train_hat, y_train)

                # Add L2 regularization
                if self.batch_m:
                    l2_reg = torch.tensor(0., requires_grad=True)
                    for name, param in self.named_parameters():
                        if name == 'reg_layer1.weight' or name == 'reg_layer1.weight':
                            l2_reg += torch.norm(param)
                    loss += 1e-8 * l2_reg

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_value += loss.item()
                
            # Get epoch loss
            epoch_loss = loss_value / (len(train_loader) * train_loader.batch_size)
            epoch_hist['train_loss'].append(epoch_loss)

            t1 = time.time()
            total_t = t1 - t0

            # Eval
            if val_loader:
                self.eval()
                val_dict = self.test_model(val_loader)
                val_loss = val_dict['loss']
                epoch_hist['valid_loss'].append(val_loss)
                print('[Epoch %d] | train_loss: %.5f | valid_loss: %.5f | Time used: %.2fs'%(epoch+1, epoch_loss, val_loss,total_t), flush=True)
            else:
                print('[Epoch %d] | train_loss: %.5f | Time used: %.2fs' % (epoch + 1, epoch_loss,total_t), flush=True)

        return epoch_hist

    def test_model(self, loader):
        """Test model on input loader."""
        test_dict = {}
        loss = 0
        loss_func = nn.CrossEntropyLoss()
        self.eval()
        with torch.no_grad():
            for x_test,y_test in loader:
                x_test = x_test.to(self.dev)
                y_test_hat = self.forward(x_test)
                loss += loss_func(y_test_hat, y_test).item()
        test_dict['loss'] = loss/(len(loader)*loader.batch_size)
        #print(y_test_hat[0][0], y_test[0][0])
        return test_dict

    def predict(self, loader):
        self.eval()
        output = []
        with torch.no_grad():
            for x_test,y_test in loader:
                x_test = x_test.to(self.dev)
                y_test_hat = self.forward(x_test)
                output.append(y_test_hat)
        return output

    def get_cell_embedding(self, bc_model=False):
        """get cell embeddings from trained model"""
        output = self.final_layer[0].weight.data.cpu()
        return output


    def get_intercept(self, bc_model=False):
        """get intercept from trained model"""
        output = self.final_layer[0].bias.data.cpu()
        return output




