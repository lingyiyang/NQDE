import math
import itertools
import numpy as np
import torch
import torchcde
import torch.nn.functional as F
from utils import get_data, save_outputs, count_parameters, modify_dict_hyperparameters

from datetime import datetime
from pathlib import Path
import yaml
from torchviz import make_dot

import random
random.seed(100)
np.random.seed(100)
torch.manual_seed(100)

import sys
sys.path.append('../')

import projunn

sampler_choice = "LSI"

# projUNN parameters
if sampler_choice == "LSI":		# column_sampling_approximation or LSI_approximation
	sampler = projunn.utils.LSI_approximation 
else:
	sampler = projunn.utils.column_sampling_approximation


# create projector and optionally the optimizer
def projector(param, update, lr_divider=4, rank=4, projector="projUNNT"):
    a, b = sampler(update/lr_divider, k=rank)
    if projector == "projUNND":
        update = projunn.utils.projUNN_D(param.data, a, b, project_on=False)
    else:
        update = projunn.utils.projUNN_T(param.data, a, b, project_on=False)
    return update


class QDEFunc(torch.nn.Module):
    '''
    QDE Function which splits then concat before linear2
    '''
    
    def __init__(self, input_channels, hidden_channels, lin_size=32):
        super(QDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, lin_size)
        self.linear2 = torch.nn.Linear(lin_size*2, input_channels * hidden_channels)
        self.rnn_layer = projunn.layers.OrthogonalRNN(lin_size, lin_size, nonlinearity = torch.nn.ReLU)
        self.rnn_layer2 = projunn.layers.OrthogonalRNN(lin_size, lin_size, nonlinearity = torch.nn.ReLU)
        self.reset_parameters()
        self.nfe = 0


    def forward(self, t, z):
        self.nfe += 1
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = z.relu()
        z, output = self.rnn_layer(z)
        z2, output2 = self.rnn_layer(z)
        z = torch.cat([z,z2], axis=1)
        z = self.linear2(z)

        z = z.tanh()
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z


    def reset_parameters(self):
        torch.nn.init.eye_(self.rnn_layer.recurrent_kernel.weight.data)
        torch.nn.init.xavier_uniform_(self.rnn_layer.input_kernel.weight.data, gain = 1.)


class QDEFunc2(torch.nn.Module):
    '''
    QDE Function which splits then concat after linear2
    '''
    
    def __init__(self, input_channels, hidden_channels, lin_size=32):

        super(QDEFunc2, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, lin_size)
        self.linear2 = torch.nn.Linear(lin_size, int(input_channels * hidden_channels/2))
        self.rnn_layer = projunn.layers.OrthogonalRNN(lin_size, lin_size, nonlinearity = torch.nn.ReLU)
        self.rnn_layer2 = projunn.layers.OrthogonalRNN(lin_size, lin_size, nonlinearity = torch.nn.ReLU)
        self.reset_parameters()
        self.nfe = 0


    def forward(self, t, z):
        self.nfe += 1
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = z.relu()
        z, output = self.rnn_layer(z)
        z2, output2 = self.rnn_layer(z)
        
        z = self.linear2(z)
        z2 = self.linear2(z2)

        z = z.tanh()
        z2 = z2.tanh()

        z = torch.cat([z,z2],axis=1)
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)

        return z


    def reset_parameters(self):
        torch.nn.init.eye_(self.rnn_layer.recurrent_kernel.weight.data)
        torch.nn.init.xavier_uniform_(self.rnn_layer.input_kernel.weight.data, gain = 1.)


class NeuralQDE(torch.nn.Module):
    def __init__(self, QDEFunc, input_channels, hidden_channels, output_channels, 
                 interpolation="cubic", collapse_type='softmax', lin_size=32, tau=1.0):
        super(NeuralQDE, self).__init__()

        self.func = QDEFunc(input_channels, hidden_channels, lin_size)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        #self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.interpolation = interpolation
        self.collapse_type = collapse_type

        if self.collapse_type == 'gumbel':   
            self.tau = tau


    def collapse(self, z):
        z_temp = z.reshape([-1,2])
        z_temp = torch.sum(z_temp**2, dim=1).reshape([z.shape[0],-1])
        return z_temp.softmax(dim=1)


    def collapse2(self, z):
        prob = self.collapse(z)
        logits = torch.log(prob)
        
        return F.gumbel_softmax(logits, tau=self.tau, hard=False)
    
    
    def forward(self, coeffs):
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")


        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)

        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.func,
                              #t=torch.tensor([0, 99.0]))
                              t=X.interval)

        z_T = z_T[:, 1]

        if self.collapse_type == 'softmax':
            prob = self.collapse(z_T)
            #print(prob)
            
        elif self.collapse_type == 'gumbel':
            prob = self.collapse2(z_T)
                
        return prob

    @property
    def nfe(self):
        return self.func.nfe

    @nfe.setter
    def nfe(self, value):
        self.func.nfe = value


def main(QDEFunc, configs):
    train_X, train_y = get_data(datasize=configs['datasize'])

    interp_key = {'cubic': 'cubic',
                  'rectilinear': 'linear',
                  'linear': 'linear',
                 }

    interp = interp_key[configs['interpolation']]
    print(interp)
    
    model = NeuralQDE(QDEFunc=QDEFunc, input_channels=3, hidden_channels=4, output_channels=1, 
                      interpolation=interp, collapse_type=configs['collapse_type'], 
                      lin_size=configs['lin_size'], tau=configs['tau'])

    print(model)
    print(f'number of trainable model parameters: {count_parameters(model)}')

    optimizer = projunn.optimizers.RMSprop(model.parameters(), projector=projector, lr=configs['lr']
    )

    scheduler = None

    if configs['lmbda'] is not None:
        lmbda = lambda epoch: configs['lmbda']
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

    if configs['interpolation'] == 'cubic':
        train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(train_X)
    elif configs['interpolation'] == 'rectilinear':
        train_coeffs = torchcde.linear_interpolation_coeffs(train_X, rectilinear=0)
    elif configs['interpolation'] == 'linear':
        train_coeffs = torchcde.linear_interpolation_coeffs(train_X)
        

    train_dataset = torch.utils.data.TensorDataset(train_coeffs, train_y)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    losses = []
    nfe_fs = []
    nfe_bs = []

    dummy_batch = next(iter(train_dataloader))
    dummy_coeffs, _ = dummy_batch
    yhat = model(dummy_coeffs)
    make_dot(yhat, params=dict(list(model.named_parameters()))).render(f"architecture_{configs['QDEFunc']}", format="eps")

    
    for epoch in range(configs['num_epochs']):
        for batch in train_dataloader:
            batch_coeffs, batch_y = batch
            pred_y = model(batch_coeffs).squeeze(-1)

            nfe_forward = model.nfe
            nfe_fs.append(nfe_forward)
            model.nfe = 0
            loss = torch.nn.functional.nll_loss(torch.log(pred_y), batch_y)
            loss.backward()
            optimizer.step()
            nfe_backward = model.nfe
            nfe_bs.append(nfe_backward)
            model.nfe = 0
            optimizer.zero_grad()
            
        if scheduler is not None and (epoch % 1) == 0:
            scheduler.step()
        print('Epoch: {}   Training loss: {}'.format(epoch, loss.item()))
        losses.append(loss.item())

    print(nfe_fs)
    print(nfe_bs)

    test_X, test_y = get_data()

    if configs['interpolation'] == 'cubic':
        test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(test_X)
    elif configs['interpolation'] == 'rectilinear':
        test_coeffs = torchcde.linear_interpolation_coeffs(test_X, rectilinear=0)
    elif configs['interpolation'] == 'linear':
        test_coeffs = torchcde.linear_interpolation_coeffs(test_X)
        
    
    pred_y = model(test_coeffs).squeeze(-1)
    binary_prediction = torch.argmax(pred_y, dim=1).to(test_y.dtype)
    prediction_matches = (binary_prediction == test_y).to(test_y.dtype)
    proportion_correct = prediction_matches.sum() / test_y.size(0)
    print('Test Accuracy: {}'.format(proportion_correct))
    
    save_outputs(configs, model, losses, proportion_correct.item(), nfe_fs, nfe_bs)


if __name__ == '__main__':

    # check for custom config
    if len(sys.argv) > 1:
        yaml_name = sys.argv[1]
    else:
        yaml_name = 'default_config.yaml'
    
    with open(Path(__file__).resolve().parent/yaml_name, 'r') as stream:
        all_configs = yaml.safe_load(stream)
    
    all_configs = modify_dict_hyperparameters(all_configs)

    funct_parser = {
        'QDEFunc': QDEFunc,
        'QDEFunc2': QDEFunc2
    }

    for configs in all_configs:
        for i in range(3):
            print(configs)
            main(funct_parser[configs['QDEFunc']], configs)
