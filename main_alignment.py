##Code used for empirically estimating the GAL for different tasks and losses.

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import random
import torch
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader
# import token_transformer
from utilities import create_test_matrix_11, calculate_fourier_coefficients
from examples import tasks
import time
import os
import models
import torch.nn.functional as F
import copy
from generate_two_cycles import generate_two_cycles

def generate_fresh_samples(batch_size, dimension,p=1/2):
    return 1 - 2 * torch.bernoulli(torch.ones((batch_size, dimension), device=device) * p)


def build_model(arch, sigma, dimension):
    if arch == 'mlp':
        model = models.MLP(input_dimension=dimension,sigma_init=sigma)
    elif arch == 'twolayermlp':
        model = models.TwoLayerMLP(input_dimension=dimension,sigma_init=sigma)
    return model.to(device)



def initial_gradient(test_X,test_y,model,loss_func,test_batch_size):
    
    model.train()  # Set model to training mode

    # Zero the gradients
    model.zero_grad()
    
    # Take a batch from the training data loader
    test_X = torch.tensor(test_X, device=device)
    test_y = torch.tensor(test_y, device=device)
    test_ds = TensorDataset(test_X, test_y)
    test_dl = DataLoader(test_ds, batch_size=test_batch_size)
    
    xb, yb = next(iter(test_dl))

    # Perform a forward pass and compute the loss
    pred = model(xb)
    loss = loss_func(pred, yb)

    # Backpropagate to compute the gradients
    loss.backward()
    gradients = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
    
    return gradients

    
    
if __name__ == '__main__':
    
    parser = ArgumentParser(description="Training script for neural networks on different functions",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    # Required runtime params
    parser.add_argument('-task', required=True, type=str, help='name of the task')
    parser.add_argument('-model', required=True, type=str, help='name of the model')
    parser.add_argument('-seed', required=True, type=int, help='random seed')
    parser.add_argument('-cuda', required=False, type=str, default='0', help='number of the gpu')
    parser.add_argument('-loss', required=False, type=str, default="", help='loss function used for training -- default is l2 while hinge, L1, corr. can also be selected.')
    parser.add_argument('-test-batch-size', type=int, default=8192, help='batch size for test samples')
    parser.add_argument('-sigma_init', default=0, type=float, help='noise level of initialization')
    parser.add_argument('-Nexp', default=10, type=int, help='nb. experiments in Montecarlo')
    parser.add_argument('-dim', default=50, type=int, help='dimension')


    args = parser.parse_args()
    start_time = time.time()
    # General setup of the experiments
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    momentum = 0.0
    

    if args.task not in tasks:
        print("Task not found.")
        exit()
    else: 
        task_params = tasks[args.task]
        dimension = task_params['dim']
        
        
    task_params.update(vars(args))
    
    test_batch_size = task_params['test_batch_size']
    sigma = task_params['sigma_init']
    Nexp = task_params['Nexp']
    
    print(vars(args))

    # Setting the seeds
    np.random.seed(task_params['seed'])
    random.seed(task_params['seed'])
    torch.manual_seed(task_params['seed'])
    
    def hinge_loss(output, target):
        return torch.max(torch.tensor(0), 1 - output * target).mean()
    
    def corr_loss(output, target):
        return (output*target).mean()
    
    def L1_loss(output, target):
        return torch.abs(target-output).mean()
    
    
    loss_func = nn.MSELoss()
    if task_params['loss'].lower() == 'hinge':
        print("Using hinge loss.")
        loss_func = hinge_loss
        
    if task_params['loss'].lower() == 'corr':
        print("Using correlation loss.")
        loss_func = corr_loss
    
    if task_params['loss'].lower() == 'l1':
        print("Using l1 loss")
        loss_func = L1_loss
    
    
    print("Computing Alignment")
    
    tot =0
    totvec = []
    
    
    
    
    
    for t in range(Nexp):
        model1 = build_model(task_params['model'], sigma, dimension)
        model2 = copy.deepcopy(model1)
        
        print("model created and copied")
        
        test_X = create_test_matrix_11(test_batch_size, dimension)
        test_y = task_params['target_function'](test_X)
            
        test_y = test_y.reshape(-1, 1)
        
        grad1 = initial_gradient(test_X,test_y,model1,loss_func,test_batch_size)
        print('Norm grad1')
        print(torch.norm(grad1,p=2))
        
        
        test_X = create_test_matrix_11(test_batch_size, dimension) 
        test_y = np.zeros(test_y.shape).astype(np.float32)
        test_y = test_y.reshape(-1, 1)
        
        grad2 = initial_gradient(test_X,test_y,model2,loss_func,test_batch_size)
        print('Norm grad2')
        print(torch.norm(grad2,p=2))
        tot = torch.norm(grad1-grad2, p=2)
        totvec.append(tot.to(torch.device('cpu')))
        print(tot)
    
    
    totvec = np.array(totvec)
    res = np.mean(totvec)
    std_res = np.std(totvec)
    print('Initial Alignment: '+str(res)+', st.dev: '+str(std_res))
    
    print("Saving data")
    
    saved_data = {'alignvec': np.array(totvec), 'align_mean':res, 'align_std':std_res}
    

    file_path = f"Alignment_{args.task}_{task_params['model']}_{'' if task_params['loss'] == '' else task_params['loss'] + '_'}{task_params['sigma_init']}_{task_params['seed']}_{task_params['dim']}_{task_params['test_batch_size']}_{task_params['Nexp']}.npz"
    
    
    
    with open(file_path, "wb") as f:
        np.savez(f, **saved_data)

