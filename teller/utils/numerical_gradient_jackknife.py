import numpy as np
from .deepcopy import deepcopy
from .memoize import memoize
from .progress_bar import Progbar
from joblib import Parallel, delayed
from tqdm import tqdm
from .numerical_gradient import numerical_gradient


@memoize
def numerical_gradient_jackknife(f, X, h=None, n_jobs=None):
    
    n, p = X.shape
    mean_grads = []
    
    
    if n_jobs is None: 
        
        print("\n")
        print("Calculating the effects...")
        pbar = Progbar(n)
        
        for i in range(n):
            
            X_i = np.delete(X, i, 0)
            
            grad_i = numerical_gradient(f, X_i, verbose=0)
            
            mean_grads.append(np.mean(grad_i, axis=0))
            
            pbar.update(i)
        
        pbar.update(n)
        print("\n")
        
        mean_grads = np.asarray(mean_grads)
        
        mean_est = np.mean(mean_grads, axis=0)
        
        se_est = ((n-1)*np.var(mean_grads, axis=0))**0.5
        
        return mean_est, se_est
    
    
    # if n_jobs is not None:    
    def gradient_column(i):
        X_i = np.delete(X, i, 0)            
        grad_i = numerical_gradient(f, X_i, verbose=0)            
        mean_grads.append(np.mean(grad_i, axis=0))
    
    print("\n")
    print("Calculating the effects...")                
    Parallel(n_jobs=n_jobs, prefer="threads")(
             delayed(gradient_column)(m)
             for m in tqdm(range(n)))
    print("\n")
        
    mean_grads = np.asarray(mean_grads)
        
    mean_est = np.mean(mean_grads, axis=0)
    
    se_est = ((n-1)*np.var(mean_grads, axis=0))**0.5
    
    return mean_est, se_est