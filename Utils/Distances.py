
import numpy as np
from keras import backend as k
import tensorflow as tf

# Maximum mean discrepancy

# Kernel functions

def radial_funtion(x,y):
    return np.linalg.norm(x-y)

def guassian_norm_funtion(x,y,sig=0.1):
    return np.exp((np.linalg.norm(x-y)**2)*(1/sig**2))

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


# Distribution Embedding

def general_normal_distribution(X):
    return np.mean(X),np.std(X)

# KL specific distances

def Univariate_Normal(P1,P2):
    mu1=P1[0]
    mu2=P2[0]
    sig1=P1[1]**2
    sig2=P2[1]**2
    return 0.5*((mu1-mu2)**2/sig2+sig1/sig2-np.log(sig1/sig2)-1)
#  KL-Divergence

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

# Mixture distribution methods

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

class Contrastive_Divergence(object):
    def __init__(self,Model):
        """
        Distribution_embedding: function with input of X dimension and output the parameters of the distribution (example mean and variance for normal distribution) 
            X_1,X_2 [Datapoints,(dimensions)]
        Distance_function: A function of distance for specific distribution with input of distribution parameters with real output
        """
        self.Model=Model
    
    def compute_layer_wise(self,X_1,X_2):

        grads_1=k.gradients(k.log(1e-5+self.Model(k.constant(X_1))),self.Model.trainable_weights)
        grads_1=np.array(list(map(lambda g:k.get_value(g),grads_1)))

        grads_2=k.gradients(k.log(1e-5+self.Model(k.constant(X_2))),self.Model.trainable_weights)
        grads_2=np.array(list(map(lambda g:k.get_value(g),grads_2)))

        multi_score=np.array(list(map(lambda x1,x2: x1-x2,grads_1,grads_2)))

        return multi_score

    def compute(self,X_1,X_2):

        multi_score=self.compute_layer_wise(X_1,X_2)

        return np.mean(np.array(list(map(lambda l:np.mean(l),multi_score))))

# Score Matching

# Noise Contrastive Estimation

# Probability flow