#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 23:43:27 2024

@author: salim
"""
import torch
import sys, os
from torch import nn
import time
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
base_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(base_path,'../models'))

#Local imports
from lstm import LSTM
from dataset import data_generator


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--dim', type=int, default=23)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--embed_dim', type=int, default=40)
parser.add_argument('--log-interval', type=int, default=10, metavar='N')
parser.add_argument('--seed', type=int, default=1111)

args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)


base_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.join(base_path,'data')
s_dir = os.path.join(base_path,'output/')

batch_size = args.batch_size
n_classes = 1  ## For Binary class
epochs = args.epochs


#device= torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cpu')

inlabels = ['Dog_1','Dog_2','Dog_3','Dog_4','Patient_1','Patient_2','Patient_3','Patient_4','Patient_5','Patient_6','Patient_7','Patient_8']

train_loader, test_loader = data_generator(0, batch_size)


data, leb = next(iter(train_loader)) # Input shape
input_shape = data.shape
print('Input shape', data.shape)
sequence_length = input_shape[-2]
dim = input_shape[-1]
embed_dim = 40
hidden_length = embed_dim

class Net(nn.Module):
    def __init__(self, sequence_length, dim, embed_dim, hidden_length):
        super().__init__()
        
        self.hidden_length = hidden_length
        self.lstm = LSTM(sequence_length, dim, embed_dim, hidden_length) #nn.LSTM(28, 28, batch_first=True)
        self.fc1 = nn.Linear(hidden_length, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h_t = torch.zeros(x.size(0), self.hidden_length).to(device)
        c_t = torch.zeros(x.size(0), self.hidden_length).to(device)


        x_t, h_t, c_t = self.lstm(x, (h_t, c_t))

        out = self.fc1(h_t)
        out = self.sigmoid(out)
        return out
    
model= Net(sequence_length, dim, embed_dim, hidden_length).to(device)


model_name = "Model_{}_dim_{}_heads_{}_lr_{}".format(
            'LSTM',sequence_length ,args.embed_dim, args.lr)

message_filename = s_dir + 'r_' + model_name + '.txt'

with open(message_filename, 'w') as out:
    out.write('start\n')


lr = args.lr

def output_s(message, save_filename):
    print (message)
    with open(save_filename, 'a') as out:
        out.write(message + '\n')
        
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr= lr)


def train(ep):
    targets = list()
    preds = list()
    train_loss = 0
    correct = 0

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()

        optimizer.step()
        train_loss += loss
        pred = output.round()
        correct += (pred== target).sum().item()
        targets += list(target.detach().cpu().numpy())
        preds += list(pred.detach().cpu().numpy())
        acc = 100. * correct / ((batch_idx+1) * batch_size)

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.2f} \t Acc: {:.2f}".format(
                ep, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loss.item()/(batch_idx),acc))

    return 100. * correct / len(train_loader.dataset), train_loss.item()/batch_size,


## Leeanable parameters counts ###
def test():
    model.eval()

    targets = list()
    preds = list()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.round()
            correct += (pred== target).sum().item()
            targets += list(target.detach().cpu().numpy())
            preds += list(pred.detach().cpu().numpy())

        Acc = 100. * correct / len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.3f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), Acc))
        return targets, preds, Acc, test_loss

#%%
if __name__ == "__main__":
    exec_time = 0
    for epoch in range(1, epochs+1):
        start = time.time()
        train_acc, train_loss = train(epoch)
        end = time.time()
        t = end-start
        exec_time+= t
        # Testing the model for each epoch
        preds, targets, test_acc, test_loss = test()
        message = ('Train Epoch: {}, Train loss: {:.4f}, Time taken: {:.4f}, Train Accuracy: {:.4f}, Test loss: {:.4f}, Test Accuracy: {:.4f}' .format(
                epoch, train_loss, t, train_acc, test_loss, test_acc))
        output_s(message, message_filename)

        if epoch % 10 == 0:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if epoch%(epochs)==0:
            print('Total Execution time for training:',exec_time)
            preds = np.array(preds)
            targets = np.array(targets)
            conf_mat= confusion_matrix(targets, preds)
            disp = ConfusionMatrixDisplay(confusion_matrix= conf_mat)
            disp.plot()
            print(classification_report(targets, preds, digits=4))


