import torch
from torch import nn
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_dimension, sigma_init,alpha_init=1/2):
        super(MLP, self).__init__()
        self.input_dimension = input_dimension
        self.sigma = sigma_init
        print(sigma_init)
        self.seq = nn.Sequential(
            nn.Linear(input_dimension, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.alpha_init = alpha_init
        self.seq.apply(self._init_weights)

    def _init_weights(self, layer):
        
        if self.sigma == -1:
            ## -1: Gaussian initialization
            if isinstance(layer, nn.Linear):
                stdv = 1. / (layer.weight.size(1))** self.alpha_init 
                # Initialize the weights with Gaussian initialization
                with torch.no_grad():
                    layer.weight.data = torch.normal(mean=0.0, std=stdv,size=layer.weight.shape)
                    layer.bias.data = torch.normal(mean=0.0, std=stdv,size=layer.bias.shape)
        
        elif self.sigma == -2:
            ## -2: Rad + small Unif perturbation (sigma = 0.1)
            if isinstance(layer, nn.Linear):
                bound = 0.1 * (3 ** 0.5) 
                stdv = 1. / (layer.weight.size(1) * (1+0.1**2))** self.alpha_init  
                
                with torch.no_grad():
                    layer.weight.data = (torch.randint(0, 2, layer.weight.shape) * 2 - 1 + (torch.rand(layer.weight.shape) * 2 * bound - bound)) * stdv    
                    layer.bias.data = (torch.randint(0, 2, layer.bias.shape) * 2 - 1 + (torch.rand(layer.bias.shape) * 2 * bound - bound)) * stdv
          
        elif self.sigma == -3:
            ## -3: Rad + large Unif perturbation (sigma = 1)
            if isinstance(layer, nn.Linear):
                bound =  (3 ** 0.5) 
                stdv = 1. / (layer.weight.size(1) * (1+1))** self.alpha_init  
                
                with torch.no_grad():
                    layer.weight.data = (torch.randint(0, 2, layer.weight.shape) * 2 - 1 + (torch.rand(layer.weight.shape) * 2 * bound - bound)) * stdv    
                    layer.bias.data = (torch.randint(0, 2, layer.bias.shape) * 2 - 1 + (torch.rand(layer.bias.shape) * 2 * bound - bound)) * stdv
                    

        elif self.sigma == -4:
            ## -4: Unif{-1,0,1} init.
            if isinstance(layer, nn.Linear):
                stdv = 1. / (layer.weight.size(1)*2/3)** self.alpha_init  
                
                with torch.no_grad():
                    layer.weight.data = (torch.randint(0, 3, layer.weight.shape) - 1) * stdv    
                    layer.bias.data = (torch.randint(0, 3, layer.bias.shape) - 1) * stdv

        elif self.sigma == -5:
            ## -5: Unif{-2-1,1,2} init.
            if isinstance(layer, nn.Linear):
                values = torch.tensor([-2, -1, 1, 2], dtype=torch.float32)
                variance = torch.var(values)
                stdv = 1. / (layer.weight.size(1)*variance)** self.alpha_init  
                
                with torch.no_grad():
                    layer.weight.data = values[torch.randint(0, 4, layer.weight.shape)]*stdv
                    layer.bias.data = values[torch.randint(0, 4, layer.bias.shape)]*stdv
                    
        elif self.sigma == -6:
            ## -6: Unif{-1,-1,0,1,1} init.
            if isinstance(layer, nn.Linear):
                values = torch.tensor([-1, -1, 0, 1,1], dtype=torch.float32)
                variance = torch.var(values)
                stdv = 1. / (layer.weight.size(1)*variance)** self.alpha_init  
                
                with torch.no_grad():
                    layer.weight.data = values[torch.randint(0, 5, layer.weight.shape)]*stdv
                    layer.bias.data = values[torch.randint(0, 5, layer.bias.shape)]*stdv  

                    
        elif self.sigma == -7:
            ## -7: Unif{-1,0,0,1} init.
            if isinstance(layer, nn.Linear):
                values = torch.tensor([-1, 0, 0, 1], dtype=torch.float32)
                variance = torch.var(values)
                stdv = 1. / (layer.weight.size(1)*variance)** self.alpha_init  
               
                with torch.no_grad():
                    layer.weight.data = values[torch.randint(0, 4, layer.weight.shape)]*stdv
                    layer.bias.data = values[torch.randint(0, 4, layer.bias.shape)]*stdv                     
                    
        else:
            # sigma non-negative: perturbed initialization
            if isinstance(layer, nn.Linear):
                stdv = 1. / (layer.weight.size(1) * (1+self.sigma**2))** self.alpha_init 
                with torch.no_grad():
                    layer.weight.data = (torch.randint(0, 2, layer.weight.shape) * 2 - 1+ torch.normal(mean=0.0, std=self.sigma,size=layer.weight.shape))*stdv
                    layer.bias.data = (torch.randint(0, 2, layer.bias.shape) * 2 - 1+ torch.normal(mean=0.0, std=self.sigma,size=layer.bias.shape))*stdv

      
    def forward(self, x):
        return self.seq(x)




class TwoLayerMLP(nn.Module):
    def __init__(self, input_dimension, sigma_init, alpha_init=1/2) -> None:
        super(TwoLayerMLP, self).__init__()
        self.input_dimension = input_dimension
        self.sigma = sigma_init
        print(sigma_init)
        self.seq = nn.Sequential(
            nn.Linear(input_dimension, 512), 
            nn.ReLU(), 
            nn.Linear(512, 1)
        )
        
        self.alpha_init = alpha_init
        self.seq.apply(self._init_weights)  # Apply the new initialization function
        

    def _init_weights(self, layer):
        
        if self.sigma == -1:
            ## -1: Gaussian initialization
            if isinstance(layer, nn.Linear):
                stdv = 1. / (layer.weight.size(1))** self.alpha_init 
                # Initialize the weights with Gaussian initialization
                with torch.no_grad():
                    layer.weight.data = torch.normal(mean=0.0, std=stdv,size=layer.weight.shape)
                    layer.bias.data = torch.normal(mean=0.0, std=stdv,size=layer.bias.shape)
        
        elif self.sigma == -2:
            ## -2: Rad + small Unif perturbation (sigma = 0.1)
            if isinstance(layer, nn.Linear):
                bound = 0.1 * (3 ** 0.5) 
                stdv = 1. / (layer.weight.size(1) * (1+0.1**2))** self.alpha_init  
                
                with torch.no_grad():
                    layer.weight.data = (torch.randint(0, 2, layer.weight.shape) * 2 - 1 + (torch.rand(layer.weight.shape) * 2 * bound - bound)) * stdv    
                    layer.bias.data = (torch.randint(0, 2, layer.bias.shape) * 2 - 1 + (torch.rand(layer.bias.shape) * 2 * bound - bound)) * stdv
          
        elif self.sigma == -3:
            ## -3: Rad + large Unif perturbation (sigma = 1)
            if isinstance(layer, nn.Linear):
                bound =  (3 ** 0.5) 
                stdv = 1. / (layer.weight.size(1) * (1+1))** self.alpha_init  
                
                with torch.no_grad():
                    layer.weight.data = (torch.randint(0, 2, layer.weight.shape) * 2 - 1 + (torch.rand(layer.weight.shape) * 2 * bound - bound)) * stdv    
                    layer.bias.data = (torch.randint(0, 2, layer.bias.shape) * 2 - 1 + (torch.rand(layer.bias.shape) * 2 * bound - bound)) * stdv
                    

        elif self.sigma == -4:
            ## -4: Unif{-1,0,1} init.
            if isinstance(layer, nn.Linear):
                stdv = 1. / (layer.weight.size(1)*2/3)** self.alpha_init  
                
                with torch.no_grad():
                    layer.weight.data = (torch.randint(0, 3, layer.weight.shape) - 1) * stdv    
                    layer.bias.data = (torch.randint(0, 3, layer.bias.shape) - 1) * stdv
        
        elif self.sigma == -5:
            ## -5: Unif{-2-1,1,2} init.
            if isinstance(layer, nn.Linear):
                values = torch.tensor([-2, -1, 1, 2], dtype=torch.float32)
                variance = torch.var(values)
                stdv = 1. / (layer.weight.size(1)*variance)** self.alpha_init  
                # Initialize the weights with Gaussian initialization
                with torch.no_grad():
                    layer.weight.data = values[torch.randint(0, 4, layer.weight.shape)]*stdv
                    layer.bias.data = values[torch.randint(0, 4, layer.bias.shape)]*stdv


        elif self.sigma == -6:
            ## -6: Unif{-1,-1,0,1,1} init.
            if isinstance(layer, nn.Linear):
                values = torch.tensor([-1, -1, 0, 1,1], dtype=torch.float32)
                variance = torch.var(values)
                stdv = 1. / (layer.weight.size(1)*variance)** self.alpha_init  
                
                with torch.no_grad():
                    layer.weight.data = values[torch.randint(0, 5, layer.weight.shape)]*stdv
                    layer.bias.data = values[torch.randint(0, 5, layer.bias.shape)]*stdv  

                    
        elif self.sigma == -7:
            ## -7: Unif{-1,0,0,1} init.
            if isinstance(layer, nn.Linear):
                values = torch.tensor([-1, 0, 0, 1], dtype=torch.float32)
                variance = torch.var(values)
                stdv = 1. / (layer.weight.size(1)*variance)** self.alpha_init  
                
                with torch.no_grad():
                    layer.weight.data = values[torch.randint(0, 4, layer.weight.shape)]*stdv
                    layer.bias.data = values[torch.randint(0, 4, layer.bias.shape)]*stdv                     
                                 
                    
        else:
            # sigma non-negative: perturbed initialization
            if isinstance(layer, nn.Linear):
                stdv = 1. / (layer.weight.size(1) * (1+self.sigma**2))** self.alpha_init 
                with torch.no_grad():
                    layer.weight.data = (torch.randint(0, 2, layer.weight.shape) * 2 - 1+ torch.normal(mean=0.0, std=self.sigma,size=layer.weight.shape))*stdv
                    layer.bias.data = (torch.randint(0, 2, layer.bias.shape) * 2 - 1+ torch.normal(mean=0.0, std=self.sigma,size=layer.bias.shape))*stdv

    def forward(self, x):
        return self.seq(x)
