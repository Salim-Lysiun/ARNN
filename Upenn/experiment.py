import torch
import sys, os
from torch import nn
import numpy as np
import argparse
sys.path.append("../../")
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import time
#Local imports
from model import Attentive_RNN
from dataset import data_generator


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--dropout', type=float, default=0.05)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--d_dim', type=int, default=23)
parser.add_argument('--embed_dim', type=int, default=40)
parser.add_argument('--dim_head', type=int, default=10)
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('--num_state_vectors', type=int, default=64)
parser.add_argument('--time_steps', type=int, default=16)
parser.add_argument('--num_class', type=int, default=1)
parser.add_argument('--qk_rmsnorm', type=bool, default=True)
parser.add_argument('--rotary_pos_emb', type=bool, default=True)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--optim', type=str, default='Adam')
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--log-interval', type=int, default=5, metavar='N')



args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)


base_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.join(base_path,'data')
s_dir = os.path.join(base_path,'output/')

batch_size = args.batch_size
n_classes = 1  ## For Binary class
epochs = args.epochs


device= torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#device = torch.device("mps")
# In[] Data Generation
inlabels = ['Dog_1','Dog_2','Dog_3','Dog_4','Patient_1','Patient_2','Patient_3','Patient_4','Patient_5','Patient_6','Patient_7','Patient_8']

train_loader, test_loader = data_generator(4, batch_size)
sig, leb = next(iter(train_loader))


heads = 4
embed_dim = 40
time_steps = 8
dim_head = int(embed_dim//heads)
input_shape = sig.shape
d_dim = input_shape[-1]
seq_len = input_shape[-2]
num_state_vectors = int(seq_len//time_steps)


# %%
model = Attentive_RNN(
    d_dim = d_dim,
    embed_dim = embed_dim,
    seq_len = seq_len,
    dim_head = dim_head,
    heads = heads,
    num_state_vectors = num_state_vectors,
    time_steps = time_steps,
    num_class=n_classes,
    qk_rmsnorm = True,
    rotary_pos_emb = True,).to(device)

model_name = "Model_{}_heads_{}_lr_{}_dropout_{}".format(
            'ARNN', args.heads, args.lr, args.dropout)

message_filename = s_dir + 'r_' + model_name + '.txt'
model_filename = s_dir + 'm_' + model_name + '.pt'
with open(message_filename, 'w') as out:
    out.write('start\n')

#%%
def output_s(message, save_filename):
    print (message)
    with open(save_filename, 'a') as out:
        out.write(message + '\n')
lr = args.lr
criterian = nn.BCELoss()
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
        loss = criterian(output, target)
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
            test_loss += criterian(output, target)
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






