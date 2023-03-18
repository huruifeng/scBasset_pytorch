import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import sys
from models.learning_utils import *

class scBasset(nn.Module):
    def __init__(self, **kwargs):
        super(scBasset, self).__init__()
        self.dev = kwargs.get('device', torch.device('cpu'))

        #self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #print(dev)
        self.num_classes = kwargs.get('num_classes', 6)
        self.width = kwargs.get('width', 1344)

        num_classes = 6
        width = 1344
        feature_layers = [] # 将卷积层存储在list中
        in_dim = 4
        out_dim = 288
        
        ## 1 cnn block
        feature_layers += [nn.Conv1d(in_channels=4,out_channels=288,kernel_size=17,padding=8), nn.MaxPool1d(kernel_size=3)]

        in_dim = 288
        ## 2 cnn tower
        for i in range(5):
            out_dim = int(np.round(in_dim * 1.122))
            feature_layers += [nn.Conv1d(in_channels=in_dim, out_channels=out_dim,kernel_size=5,padding=2), nn.ReLU(inplace = True)]
            in_dim = out_dim
            feature_layers += [nn.MaxPool1d(2)]
            self.features = nn.Sequential(*feature_layers) 

        ##
        feature_layers += [nn.Conv1d(in_channels=512,out_channels=256,kernel_size=1), nn.MaxPool1d(kernel_size=2)]
        self.features = nn.Sequential(*feature_layers)

        feature_layers += [nn.Conv1d(in_channels=256,out_channels=32,kernel_size=1), nn.MaxPool1d(kernel_size=3)]
        self.features = nn.Sequential(*feature_layers)

        self.classifier = nn.Sequential(
          nn.Linear(64, 64),
          nn.ReLU(inplace = True),
          nn.Dropout(),
          nn.Linear(64,num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        output = self.classifier(x)
        return output

    def train_model(self, train_loader, learning_rate, n_epochs, train_patience, test_patience, test_loader=False, save_model=True):
        """ Train NaiveCNN """
        epoch_hist = {}
        epoch_hist['train_loss'] = []
        epoch_hist['valid_loss'] = []
        ### old: optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=5e-4)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_func = nn.CrossEntropyLoss()
        train_ES = EarlyStopping(patience=train_patience, verbose=True, mode='train')
        if test_loader:
            valid_ES = EarlyStopping(patience=test_patience, verbose=True, mode='valid')
        # Train
        for epoch in range(n_epochs):
            loss_value = 0
            self.train()
            for x_train,y_train in train_loader:
                x_train = x_train.to(self.dev)
                y_train = y_train.to(self.dev)
                optimizer.zero_grad()
                y_train_hat = self.forward(x_train)
                #print(y_train[0], y_train_hat[0])
                loss = loss_func(y_train_hat, y_train)
                loss.backward()
                optimizer.step() 
                loss_value += loss.item()
                
            # Get epoch loss
            epoch_loss = loss_value / (len(train_loader) * train_loader.batch_size)
            epoch_hist['train_loss'].append(epoch_loss)
            train_ES(epoch_loss)
            # Eval
            if test_loader:
                self.eval()
                test_dict = self.test_model(test_loader)
                test_loss = test_dict['loss']
                epoch_hist['valid_loss'].append(test_loss)
                valid_ES(test_loss)
                print('[Epoch %d] | loss: %.5f | test_loss: %.5f |'%(epoch+1, epoch_loss, test_loss), flush=True)
                if valid_ES.early_stop or train_ES.early_stop:
                    print('[Epoch %d] Early stopping' % (epoch+1), flush=True)
                    break
            else:
                print('[Epoch %d] | loss: %.5f |' % (epoch + 1, epoch_loss), flush=True)
                if train_ES.early_stop:
                    print('[Epoch %d] Early stopping' % (epoch+1), flush=True)
                    break
        # Save model
        if save_model:
            print('Saving model to ...', self.save_path)
            torch.save(self.state_dict(), self.save_path)

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



