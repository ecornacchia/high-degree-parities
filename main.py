##This is the file containing the main code used to run the experiments

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import random
import torch
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader
from utilities import create_test_matrix_11
from examples import tasks
import time
import os
import models
import torch.nn.functional as F

def generate_fresh_samples(batch_size, dimension,p=1/2):
    return 1 - 2 * torch.bernoulli(torch.ones((batch_size, dimension), device=device) * p)


def build_model(arch, sigma, dimension):

    if arch == 'mlp':
        model = models.MLP(input_dimension=dimension,sigma_init=sigma)
    elif arch == 'twolayermlp':
        model = models.TwoLayerMLP(input_dimension=dimension,sigma_init=sigma)
    return model.to(device)




def train(train_X, train_y, valid_X, valid_y, test_X, test_y, eps, sigma, Tmax, computation_interval=0, verbose_interval=0, model=None):
     if model is None:
        print("Creating model with sigma="+str(sigma))
        model = build_model(task_params['model'], sigma, dimension)
        print("model created")
    iter_logs = []
    train_losses = []
    valid_losses = []
    test_losses = []
    train_accs = []
    valid_accs = []
    test_accs = []

    # Preparing the dataset
    if train_X is not None:
        train_y = train_y.reshape(-1, 1)
    valid_y = valid_y.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)

    if train_X is not None:
        train_X = torch.tensor(train_X, device=device)
        train_y = torch.tensor(train_y, device=device)
        
    valid_X = torch.tensor(valid_X, device=device)
    valid_y = torch.tensor(valid_y, device=device)
    test_X = torch.tensor(test_X, device=device)
    test_y = torch.tensor(test_y, device=device)

    if train_X is not None:
        train_ds = TensorDataset(train_X, train_y)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_ds = TensorDataset(valid_X, valid_y)
    valid_dl = DataLoader(valid_ds, batch_size=test_batch_size)
    test_ds = TensorDataset(test_X, test_y)
    test_dl = DataLoader(test_ds, batch_size=test_batch_size)

    # Defining the optimizer
    if task_params['opt'].lower() == 'sgd':
        print("Using SGD")
        opt = optim.SGD(model.parameters(), lr=task_params['lr'], momentum=momentum, weight_decay=0.0)
    else:
        print("Non defined optimizer")
    
    def hinge_loss(output, target):
        return torch.max(torch.tensor(0), 1 - output * target).mean()


    loss_func = nn.MSELoss()
    if task_params['loss'].lower() == 'hinge':
        print("Using hinge loss.")
        loss_func = hinge_loss
    

    
    # Function used for evaluation of the model, i.e., calculation of coefficients and valid/test losses. 
    def model_evaluation(iter, train_loss, train_acc):
        model.eval()
        with torch.no_grad():
            valid_loss = 0
            valid_acc = 0
            for xb, yb in valid_dl:
                pred = model(xb)
                valid_loss += loss_func(pred, yb)
                valid_acc += ((pred.sign() * yb) + 1).sum() / 2
            valid_loss /= len(valid_dl)
            valid_acc /= len(valid_y)

            test_loss = 0
            test_acc = 0
            for xb, yb in test_dl:
                pred = model(xb)
                test_loss += loss_func(pred, yb)
                test_acc += ((pred.sign() * yb) + 1).sum() / 2
            test_loss /= len(test_dl) 
            test_acc /= len(test_y)

            if train_loss is None:
                train_loss = valid_loss
                train_acc = valid_acc

            train_loss = train_loss.cpu().detach().numpy()
            valid_loss = valid_loss.cpu().detach().numpy()
            test_loss = test_loss.cpu().detach().numpy()
            train_acc = train_acc.cpu().detach().numpy()
            valid_acc = valid_acc.cpu().detach().numpy()
            test_acc = test_acc.cpu().detach().numpy()

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            valid_accs.append(valid_acc)
            test_accs.append(test_acc)
            iter_logs.append(iter)

            if (iter % verbose_interval == 0) or (train_loss < eps):
                print(f"Iter: {iter:8}, Train Loss: {train_loss:0.6}, Valid Loss: {valid_loss:0.6}, Test Loss: {test_loss:0.6}, Train Acc: {train_acc:0.3}, Valid Acc: {valid_acc:0.3}, Test Acc: {test_acc:0.3}, Elapsed Time:", time.strftime("%H:%M:%S",time.gmtime(time.time() - start_time)))

    
    # Model's evaluation before training
    model_evaluation(0, None, None)
    iter_counter = 0
    train_loss = torch.tensor(0.0, device=device)
    train_acc = 0
    model.train()
    training_flag = True
    while training_flag:
        if train_X is not None:
            for xb, yb in train_dl:
                pred = model(xb)
                loss = loss_func(pred, yb)
                train_loss += loss
                train_acc += ((pred.sign() * yb) + 1).mean() / 2
                loss.backward()
                opt.step()
                opt.zero_grad()
                iter_counter += 1
                if iter_counter % computation_interval == 0:
                    train_loss /= computation_interval
                    train_acc /= computation_interval
                    model_evaluation(iter_counter, train_loss, train_acc)
                    if train_loss < eps or iter_counter>Tmax:
                        training_flag = False
                        break
                    model.train()
                    train_loss *= 0
                    train_acc *= 0
                 
        else:
            xb = generate_fresh_samples(batch_size, dimension if train_y==1 else 1.0)
            yb = task_params['target_function'](xb).reshape(-1, 1)
            pred = model(xb)
            loss = loss_func(pred, yb)
            train_loss += loss
            train_acc += ((pred.sign() * yb) + 1).mean() / 2
            loss.backward()
            opt.step()
            opt.zero_grad()
            iter_counter += 1
            if iter_counter % computation_interval == 0:
                train_loss /= computation_interval
                train_acc /= computation_interval
                model_evaluation(iter_counter, train_loss, train_acc)
                if train_loss < eps or iter_counter>Tmax:
                    training_flag = False
                    break
                model.train()
                train_loss *= 0
                train_acc *= 0

            
    return iter_logs, train_losses, valid_losses, test_losses, train_accs, valid_accs, test_accs, coefficients, model


if __name__ == '__main__':
    
    parser = ArgumentParser(description="Training script for neural networks on different functions",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    # Required runtime params
    parser.add_argument('-task', required=True, type=str, help='name of the task')
    parser.add_argument('-model', required=True, type=str, help='name of the model')
    parser.add_argument('-lr', required=True, type=float, help='learning rate')
    parser.add_argument('-seed', required=True, type=int, help='random seed')
    parser.add_argument('-train-size', required=True, type=int, help='the size of the training set')
    # Other runtime params
    parser.add_argument('-cuda', required=False, type=str, default='0', help='number of the gpu')
    parser.add_argument('-eps', required=False, type=float, default=0.00001, help='threshold to stop')
    parser.add_argument('-loss', required=False, type=str, default="", help='loss function used for training -- default is l2 while hinge can also be selected.')
    parser.add_argument('-opt', default='sgd', type=str, help='sgd')
    parser.add_argument('-batch-size', default=64, type=int, help='batch size')
    parser.add_argument('-test-batch-size', type=int, default=8192, help='batch size for test samples')
    parser.add_argument('-verbose-int', default=100, type=int, help="the interval between prints")
    parser.add_argument('-compute-int', default=100, type=int, help="the interval between computations of losses")
    parser.add_argument('-sigma_init', default=0, type=float, help='noise level of initialization')
    parser.add_argument('-Tmax', default=1000000, type=float, help='max nb training epochs before breaking')
    
    args = parser.parse_args()
    start_time = time.time()
    
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    momentum = 0.0

    if args.task not in tasks:
        print("Task not found.")
        exit()
    task_params = tasks[args.task]
    dimension = task_params['dimension']
    task_params.update(vars(args))
    batch_size = task_params['batch_size']
    test_batch_size = task_params['test_batch_size']
    eps =task_params['eps']
    sigma = task_params['sigma_init']
    Tmax = task_params['Tmax']
    

    print(vars(args))

    # Setting the seeds
    np.random.seed(task_params['seed'])
    random.seed(task_params['seed'])
    torch.manual_seed(task_params['seed'])

    if task_params['train_size'] > 0:
        # Generating train, valid, and test data. We use num_samples = 0 for fresh batches. 
        train_X = create_test_matrix_11(task_params['train_size'], dimension)
        train_y = task_params['target_function'](train_X)

    else:
        train_X = None
        train_y = 1

    valid_X = create_test_matrix_11(task_params['test_size'], dimension)
    valid_y = task_params['target_function'](valid_X)
  
    test_X = create_test_matrix_11(task_params['test_size'], dimension) 
    test_y = task_params['target_function'](test_X)


    print("Starting training")
    iter_logs, train_losses, valid_losses, test_losses, train_accs, valid_accs, test_accs, coefficients,model = train(train_X, train_y, valid_X, valid_y, test_X, test_y, eps, sigma, Tmax, computation_interval=task_params['compute_int'], verbose_interval=task_params['verbose_int'])
    
    print("Saving data")
    saved_data = {'iters': np.array(iter_logs), 'train_losses': train_losses, 'valid_losses': valid_losses, 'test_losses': test_losses, 'train_accs': train_accs, 'valid_accs': valid_accs, 'test_accs': test_accs, 'run_params': vars(args), 'sigma_init': sigma, 'model':model.to(torch.device('cpu'))}

    
    file_path = f"{args.task}_{task_params['model']}_{'' if task_params['loss'] == '' else task_params['loss'] + '_'}{task_params['sigma_init']}_{task_params['seed']}_{task_params['lr']}_{task_params['opt']}_{task_params['train_size']}.npz"
    
    
    
    with open(file_path, "wb") as f:
        np.savez(f, **saved_data)

