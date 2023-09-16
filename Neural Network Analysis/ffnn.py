'''
Description of the program:
---------------------------
In this program a feedforward neutal network is implemented using pytorch to estimate the critical temperature of the phase transition of the ferromagnetic Ising Model on the triangular lattice.
Hereby, the network is trained on labeled datapoints on the square lattice (supervised learning). The network solves the binary classification task, whether a shown sample (i.e. spin configuration of the lattice) belongs to
the paramagnetic or the ferromagnetic phase. The data points are snapshots of the spin configuration on a lattice saved during the run of Monte Carlo simulations with the Wolff algorithm.
The network implemented works for a 30 times 30 lattice, since it has 900 input neurons.

Required data:
--------------
Dataset for training, dataset for testing and dataset for transfer.

The first two datasets are snapshots of the spin configurations on the square lattice, while the last one consists of snapshots of spin configurations on the triangular lattice.
The datapoints of each dataset consist of the spins and a temperature label which enables to determine whether the spin configuration belongs to the paramagnetic or the ferromagnetic phase.
In the training process, the labels are passed to the network as well. In the testing and transfer runs, the labels are not passed to the network since the network is supposed to return a classification output.
However, the labels of the test and transfer dataset are used to evaluate the accuracy of the network.

There is a dummy snapshot in the directory, as well, from which the structure can be seen. The snapshots were generated with the Wolff algorithm. This also integrated in the Wolff implementation as given in the repository.

Information on neural networks are available in the "Deep Learning" book by Goodfellow et al.:  https://www.deeplearningbook.org

'''


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import glob
import os
import csv
import numpy as np
import math
import sys
import pandas as pd

# Set up root dir for the training data (i.e. where the snapshot csv files are stored).
root_dir=''

# Configurations to control whether the network is loaded and/or tested
default=True
loading=default
testing=default


# Some global parameters:
input_size = 900 # for a 30 times 30 square lattice or triangular lattice
hidden_size = 8 # number of neurons on the single hidden layer
num_classes = 1 # binary classification
num_epochs = 250 # number of iterations
batch_size = 32
learning_rate = 0.01 # step of the optimization

# Saving or loading the model as ".pt" file in the "models_single_layer" subdirectory.
save_load_model='models_single_layer/ffnn_ising_model'+'_hiddensize_'+str(hidden_size)+'_num_epochs_'+str(num_epochs)+'_batch_size_'+str(batch_size)+'_learning_rate_'+str(learning_rate)+'_'+'.pt'

# Used for the snapshots that were generated on the square lattice. Separating training data from testing data.
class Snapshots(Dataset):
    def __init__(self,bool_isTrain):
        # load the data
        if (bool_isTrain==True):
            # If there are different datasets in the directory, they can be labelled by integers. In this case they are in subdirectories named by the specific number, e.g. "1", "2" or "3".
            # THIS HAS TO BE ALTERED DEPENDING ON THE NUMBER OF DATASETS AND NAMING CONVENTIONS.
            sim_no=[1,2,3,...]
        no_sim_no=len(sim_no)
        x=torch.zeros((1,900), dtype=torch.float32)  # datapoints
        y=torch.zeros((1,1), dtype=torch.float32)    # corresponding lable for supervised learning
        beta_tensor=torch.zeros((1,1), dtype=torch.float32)

        for i in range(0,no_sim_no):    # Loop through the simulated datasets.
            path=root_dir+str(sim_no[i])+'/' # This is the path where all the relevant snapshot csv files are.
            filenames=[]

            # Create a list of snapshot csv files to be loaded:
            for root, sub, files in os.walk(path):
                for filename in files:
                    if (filename.endswith('.csv') and filename.find('snap_lattice')!=-1):
                        filenames.append(os.path.join(root,filename))

            # Load the snapshot csv files and put them into a flattened tensor:
            for file in filenames:
                # Read the beta value of the snapshot:
                with open(file, newline='') as f:
                    reader = csv.reader(f)
                    beta = next(reader) # Gets the first line.
                    beta = float(beta[0])

                # Read all the data lines, flatten them: '[beta, 1, 1, -1 ,...]'
                # THIS MUST BE ALTERED IF A LATTICE IS USED WITH DIFFERENT NUMBER OF SITES SINCE THIS REFERS TO A 30X30 LATTICE MATRIX.
                tmp=np.loadtxt(file,delimiter=';', usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29), skiprows=1, dtype=np.float32)
                tmp=tmp.flatten()

                # Setting up the label depending on the beta value:
                beta_val=torch.tensor([beta], dtype=torch.float32)
                if (beta<0.4407):
                    beta=0.0
                else:
                    beta=1.0

                beta=torch.tensor([beta], dtype=torch.float32)
                beta=torch.unsqueeze(beta,dim=0)
                beta_val=torch.unsqueeze(beta_val, dim=0)
                tmp=torch.from_numpy(tmp)
                tmp=torch.unsqueeze(tmp, dim=0)
                # x stores the vector of dimension 900 holding the spins.
                x=torch.cat((x,tmp),0)
                # y stores the label, i.e. 0 (paramagnetic) and 1 (ferromagnetic).
                y=torch.cat((y,beta),0)
                beta_tensor=torch.cat((beta_tensor,beta_val),0)

        print(f'x shape {x.shape}')
        print(f'y shape {y.shape}')
        self.isTrain=bool_isTrain
        self.x_data=x[1:,:]
        self.y_data=y[1:,:]
        self.beta_data=beta_tensor[1:,:]
        self.n_samples=x.shape[0]-1
    def __getitem__(self, i):
        #get one specific item of the data
        return self.x_data[i], self.y_data[i], self.beta_data
    def __len__(self):
        #return the length
        return self.n_samples

# Used for loading the data that were generated for the triangular lattice.
class Snapshots_Test(Dataset):
    def __init__(self,lattice_type):
        #Load the data depending whether the testing dataset is a triangular or a square lattice dataset.
        # THIS HAS TO BE ALTERED DEPENDING ON THE NUMBER OF DATASETS AND NAMING CONVENTIONS.
        if lattice_type=='triangular':
            sim_no=[1,2,3,4,5,6]    # for the transfer # THIS HAS TO BE ALTERED DEPENDING ON THE NUMBER OF DATASETS AND NAMING CONVENTIONS.
            # Setup root directory where the testing data (i.e. the snapshot csv files) are stored for the triangular lattice.
            root_dir=''
        if lattice_type=='square':
            sim_no=[9,11]        # for the testing # THIS HAS TO BE ALTERED DEPENDING ON THE NUMBER OF DATASETS AND NAMING CONVENTIONS.
            # Setup root directory where the testing data (i.e. the snapshot csv files) are stored for the square lattice.
            root_dir=''
        no_sim_no=len(sim_no)
        x=torch.zeros((1,900), dtype=torch.float32)  #datapoints
        y=torch.zeros((1,1), dtype=torch.float32)    #lable
        beta_tensor=torch.zeros((1,1), dtype=torch.float32)

        for i in range(0,no_sim_no):    # Loop through the  simulated datasets.
            path=root_dir+str(sim_no[i])+'/'#This is the path where all the relevant snapshot csv files are.
            filenames=[]

            # Create a list of files ot be loaded:
            for root, sub, files in os.walk(path):
                for filename in files:
                    if (filename.endswith('.csv') and filename.find('snap_lattice')!=-1):
                        filenames.append(os.path.join(root,filename))
            print(len(filenames))

            # Load files and put them into a flattened tensor:
            for file in filenames:
                # Read the beta value of the snapshot:
                with open(file, newline='') as f:
                    reader = csv.reader(f)
                    beta = next(reader) # gets the first line
                    beta = float(beta[0])

                # Read all the data lines, flatten them: '[beta, 1, 1, -1 ,...]'
                tmp1=np.genfromtxt(file, delimiter=';', skip_header=1)
                if lattice_type=='square':
                # THIS MUST BE ALTERED IF A LATTICE IS USED WITH DIFFERENT NUMBER OF SITES SINCE THIS REFERS TO A 30X30 LATTICE MATRIX.
                    tmp=np.loadtxt(file,delimiter=';', usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29), skiprows=1, dtype=np.float32)
                    tmp=tmp.flatten()
                else: #i.e. triangular
                    tmp=tmp1[:900].astype(np.int32)

                beta_val=torch.tensor([beta], dtype=torch.float32)
                #beta_sq=0.4407
                #beta_tri=1/3.641
                # Labels depending on whether this is a triangular or a square lattice dataset:
                if lattice_type=='triangular':
                    if (beta<(1/3.641)):
                        beta=0.0
                    else:
                        beta=1.0
                if lattice_type=='square':
                    if (beta<(0.4407)):
                        beta=0.0
                    else:
                        beta=1.0
                beta=torch.tensor([beta], dtype=torch.float32)
                beta=torch.unsqueeze(beta,dim=0)
                beta_val=torch.unsqueeze(beta_val, dim=0)
                tmp=torch.from_numpy(tmp)
                tmp=torch.unsqueeze(tmp, dim=0)
                # x stores the vector of dimension 900 holding the spins.
                x=torch.cat((x,tmp),0)
                # y stores the label, i.e. 0 (paramagnetic) and 1 (ferromagnetic).
                y=torch.cat((y,beta),0)
                beta_tensor=torch.cat((beta_tensor,beta_val),0)

        print(f'x shape {x.shape}')
        print(f'y shape {y.shape}')
        self.x_data=x[1:,:]
        self.y_data=y[1:,:]
        self.beta_data=beta_tensor[1:,:]
        self.n_samples=x.shape[0]-1

    def __getitem__(self, i):
        #get one specific item of the data
        return self.x_data[i], self.y_data[i], self.beta_data[i]
    def __len__(self):
        #return the length
        return self.n_samples

# Check whether the model should be loaded. If not, use the data loader to prepare the datasets to be shown to the network.
if loading==False:
    train_ds=Snapshots(True)
    train_loader=DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=0)


# Create the testing and transfer datasets to be shown to the network.
if testing==True:
    test_ds=Snapshots(False)
    test_loader=DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    test_triang_ds=Snapshots_Test('triangular')
    test_triang_loader=DataLoader(dataset=test_triang_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    test_square_ds=Snapshots_Test('square')
    test_square_loader=DataLoader(dataset=test_square_ds, batch_size=batch_size, shuffle=False, num_workers=0)




# Constructing a ffnn using pytorch for a binary classification task
# nn model inheritting from torch.nn
class NeuralNet(nn.Module):
    # constructor
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    # forward method
    def forward(self, x):
        out = self.l1(x)
        #activation function
        out = self.relu(out)
        out = self.l2(out)
        # activation function returning a real number in [0,1] to be interpreted as probability
        out = torch.sigmoid(out)
        return out

# Create an instance of the NeuralNet class.
model = NeuralNet(input_size, hidden_size, num_classes)

# Use the BCE loss for a binary classification task.
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


#LOADING:
# Load the previously trained model depending on the run parameters defined at the beginning of this program.
if loading==True:
    model = NeuralNet(input_size, hidden_size, num_classes)
    model.load_state_dict(torch.load(save_load_model))
    model.eval()

#TRAINING:
# Train the model depending on the run parameters defined at the beginning of this program.
else:
    n_total_steps = len(train_loader)
    print('Calculating the weights: ')
    for epoch in range(num_epochs):
        for i, (x, y, beta) in enumerate(train_loader):
            outputs = model(x)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.7f}')
    # Save the trained model.
    torch.save(model.state_dict(),save_load_model)

#TESTING:
# Testing loop for square lattice
if testing==True:
    with torch.no_grad():

        n_correct = 0
        n_samples = 0

        # Testing on the square lattice:
        df=pd.DataFrame(columns=['y', 'output', 'classification'])
        j=0
        for x, y, beta in test_square_loader:
            outputs = model(x)
            np_y=y.numpy()
            np_outputs=outputs.numpy()
            outputs_rounded=outputs.round()
            np_beta=beta.numpy()
            np_outputs_rounded=outputs_rounded.numpy()
            for i in range(0, np_y.shape[0]):
                d=[np_beta[i][0],np_y[i][0], np_outputs[i][0], np_outputs_rounded[i][0]]
                # Store the output of the network for the testing data:
                tmp=pd.DataFrame(data=[d],columns=['beta','y', 'output', 'classification'])
                df=pd.concat([df,tmp],sort=False)

            _, predicted = torch.max(outputs.data, 1)
            n_samples += y.size(0)
            n_correct += outputs_rounded.eq(y).sum()

        # Calculate the accuracy of the trained model:
        acc = 100.0 * n_correct / n_samples
        acc_testing=acc
        df.to_csv('models_single_layer/nn_res_square_30x30_hiddens_'+str(hidden_size)+'_batch_size_'+str(batch_size)+'_num_epochs_'+str(num_epochs)+'_learning_rate_'+str(learning_rate)+'.csv', sep=';', encoding='utf-8',  header=True)
        print(f'Accuracy of the network for square lattice testing data: {acc} %')
        print('----------')
        n_correct = 0
        n_samples = 0


        # Testing on the triangular lattice:
        print('Testing on the triangular lattice:')
        df=pd.DataFrame(columns=['y', 'output', 'classification'])
        j=0
        for x, y, beta in test_triang_loader:
            outputs = model(x)
            np_y=y.numpy()
            np_outputs=outputs.numpy()
            outputs_rounded=outputs.round()
            np_beta=beta.numpy()
            np_outputs_rounded=outputs_rounded.numpy()
            for i in range(0, np_y.shape[0]):
                d=[np_beta[i][0],np_y[i][0], np_outputs[i][0], np_outputs_rounded[i][0]]
                # Store the output of the network for the testing data:
                tmp=pd.DataFrame(data=[d],columns=['beta','y', 'output', 'classification'])
                df=pd.concat([df,tmp],sort=False)

            _, predicted = torch.max(outputs.data, 1)
            n_samples += y.size(0)
            n_correct += outputs_rounded.eq(y).sum()

        # Calculate the accuracy of the trained model:
        acc = 100.0 * n_correct / n_samples
        acc_transfer=acc
        df.to_csv('models_single_layer/nn_res_triangular_30x30_hiddens_'+str(hidden_size)+'_batch_size_'+str(batch_size)+'_num_epochs_'+str(num_epochs)+'_learning_rate_'+str(learning_rate)+'.csv', sep=';', encoding='utf-8',  header=True)
        print(f'Accuracy of the network for triangular lattice testing data: {acc} %')
        print(df)
        with open('models_single_layer/meta_nn_hidden_size.txt', 'a') as f:
            f.write(str(hidden_size)+';'+str(acc_validation.item())+';'+str(acc_testing.item())+';'+str(acc_transfer.item())+';'+str(learning_rate)+';'+str(num_epochs)+';'+str(batch_size)+'\n')
