##This is the code for estimating the GAL on a 1 neuron network with correlation loss (Figure 2(right) and Figure 6)


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import math
from scipy.integrate import quad
from scipy.special import erf,erfc
from scipy.stats import norm
import time

def prob(k,n,m,d):
    return math.comb(m,k)*math.comb(d-m,n-k)*math.comb(d,m)*4**(-d)

def myf(g,c,s,sigma):
    return math.exp(-g**2/2)*(1+erf((c*g+s)/math.sqrt(max(2-2*c**2,1e-55))))/(math.sqrt(2*math.pi)*2)

def run(sigma,d):
    tot = 0
    
    for n in range(d+1):
        for m in range(d+1):
            for k in range(min(n,m)+1):
                p = prob(k,n,m,d)

                s = (d-2*n)/(sigma*math.sqrt(d))
                t = (d-2*m)/(sigma*math.sqrt(d))
                c = (d-2*n-2*m+4*k)/d
                sign = (-1)**((m+n)%2)
                
                if p>0:
                    if abs(c)<1:
                        integral, error = quad(lambda g: myf(g,c,s,sigma),-t,100)
                        tot +=p*integral*sign
                    
                    if c==1:
                        integral = 1/2*(1-erf(max(-s/np.sqrt(2),-t/np.sqrt(2))))
                        tot +=p*integral*sign
                        
                    if c==-1 and -s<t:
                        integral = 1/2*(-erf(-s/np.sqrt(2)))+1/2*erf(t/np.sqrt(2))
                        tot +=p*integral*sign
    return tot



if __name__ == "__main__":
    parser = ArgumentParser(description="Script for computing INAL for parity with mixed initialization",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    
    # Required runtime params
    parser.add_argument('-d', required=True, type=int, help='input dimension')
    parser.add_argument('-sigma', required=True, type=float, help='st. dev. of initial mixed distr.')
    
    args = parser.parse_args()
    start_time = time.time()
    
    d = args.d
    sigma = args.sigma
    
    res = run(sigma,d)
    
    print('d='+str(d)+', sigma='+str(sigma)+', INAL='+str(res))
    
    data_to_save = {'d': d, 'sigma': sigma, 'INAL': res}
    
    with open(f"{d}_{sigma}.npz","wb") as f:
        np.savez(f, **data_to_save)

        
        


    
