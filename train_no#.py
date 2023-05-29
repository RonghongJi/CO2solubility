import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
#from models.gat import GATNet
#from models.gat_gcn import GAT_GCN
from gcn_def import GCNNet
#from models.ginconv import GINConvNet
from utils import *
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score 


os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()

# modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][int(sys.argv[2])]
modeling = GCNNet
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv)>3:
    cuda_name = "cuda:" + str(int(sys.argv[3])) 
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets


processed_data_file_train = '/home/rhji/CO2_solubility/data/processed/train_jrh_new.pt'
processed_data_file_test = '/home/rhji/CO2_solubility/data/processed/test_jrh_new.pt'
train_data = TestbedDataset(root='data', dataset='train_jrh_new')
test_data = TestbedDataset(root='data', dataset='test_jrh_new')
train_data = train_data[0:7000]
test_data = test_data[0:3000]
        
# make data PyTorch mini-batch processing ready
train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

# training the model
device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
model = modeling().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
best_mse = 10
best_ci = 0
best_epoch = -1
model_file_name = 'model_' + model_st  +  '.model'
result_file_name = 'result_' + model_st +  '.csv'

for epoch in range(NUM_EPOCHS):
    train(model, device, train_loader, optimizer, epoch+1)

G,P = predicting(model, device, test_loader)
ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P)]
if ret[1]<best_mse:
    torch.save(model.state_dict(), model_file_name)
    with open(result_file_name,'w') as f:
        f.write(','.join(map(str,ret)))
    best_epoch = epoch+1
    best_mse = ret[1]
    best_ci = ret[-1]
    print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse,best_ci,model_st)
else:
    print(ret[1],'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse,best_ci,model_st)
# a = mean_squared_error(G,P)
# b = mean_absolute_error(G,P)
# c = r2_score(G,P)
# print('test_RMSE:'+ str(a))
# print('test_MAE:'+ str(b))
# print('test_R^2:'+ str(c))

# G_1, P_1 = predicting(model, device, train_loader)
# a_1 = mean_squared_error(G_1,P_1)
# b_1 = mean_absolute_error(G_1,P_1)
# c_1 = r2_score(G_1,P_1)
# print('train_RMSE:'+ str(a_1))
# print('train_MAE:'+ str(b_1))
# print('train_R^2:'+ str(c_1))
