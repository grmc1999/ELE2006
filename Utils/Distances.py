
import numpy as np

# Maximum mean discrepancy

# Kernel functions

def radial_funtion(x,y):
    return np.linalg.norm(x-y)

def guassian_norm_funtion(x,y,sig=0.1):
    return (np.linalg.norm(x-y)**2)*(1/sig**2)

# class
class MMD(object):
    def __init__(self,kernel,kernel_trick=False):
        """
        kernel: function of x and y to scalar value
        kernel_trick: function with x,y input and xx,xy,yy output
        """
        self.kernel_trick=kernel_trick
        self.kernel=kernel
    def compute(self,x,y):
        """
        x: vector [Data points,data dimension]
        y: vector [Data points,data dimension]
        """
        if self.kernel_trick:
            xx,xy,yy=self.kernel_trick(x,y)
        else:
            xx=np.array(list(map(lambda x,y: self.kernel(x,y),x,x)))
            xy=np.array(list(map(lambda x,y: self.kernel(x,y),x,y)))
            yy=np.array(list(map(lambda x,y: self.kernel(x,y),x,x)))
        
        return np.mean(xx-2*xy+yy)

def general_normal_distribution(X):
    return np.mean(X),np.std(X)


# KL-Divergence

class KL_divergence(object):
    def __init__(self,Distribution_embedding,Distance_function):
        """
        Distribution_embedding: function with input of X dimension and output the parameters of the distribution (example mean and variance for normal distribution) 
            X [Datapoints,(dimensions)]
        Distance_function: A function of distance for specific distribution with input of distribution parameters with real output
        """
        self.Distribution_embedding=Distribution_embedding
        self.Distance_function=Distance_function
    
    def compute(self,X_1,X_2):
        D1_param=self.Distribution_embedding(X_1)
        D2_Param=self.Distribution_embedding(X_2)
        distance=self.Distance_function(D1_param,D2_Param)
        return distance


# Jensen-Shanon Divergence

def median_gaussian(d1,d2):
    return (d1[0]+d2[0])/2,(d1[1]+d2[1])/2

class Jensen_Shanon_divergence(object):
    def __init__(self,Distribution_embedding,Distance_function,Mixture_distribution_method):
        """
        Distribution_embedding: function with input of X dimension and output the parameters of the distribution (example mean and variance for normal distribution) 
            X_1,X_2 [Datapoints,(dimensions)]
        Distance_function: A function of distance for specific distribution with input of distribution parameters with real output
        """
        self.Distribution_embedding=Distribution_embedding
        self.Distance_function=Distance_function
        #self.KLD=KL_divergence(self.Distribution_embedding,self.Distance_function)
        self.Mixture_distribution_method=Mixture_distribution_method
    
    def compute(self,X_1,X_2):
        D1_param=self.Distribution_embedding(X_1)
        D2_param=self.Distribution_embedding(X_2)
        D12_param=self.Mixture_distribution_method(D1_param,D2_param)
        distance=(self.Distance_function(D1_param,D12_param)+self.Distance_function(D2_param,D12_param))/2

        return distance

# Contrastive Divergence

# Score Matching

# Noise Contrastive Estimation

# Probability flow