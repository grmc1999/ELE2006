
import numpy as np
from einops import rearrange,repeat
from keras import backend as k
import tensorflow as tf
from glob import glob
import os
from scipy.stats import entropy


class distance_base(object):

    def no_strategy(self,dataA,dataB):
        return dataA,dataB

    def over_sample_smaller(self,dataA,dataB):
        size_diff=len(dataA)-len(dataB)
        oversample=(lambda data,size_diff:(data+np.random.choice(data,abs(size_diff),replace=False).tolist()))
        return (oversample(dataA,size_diff),dataB) if size_diff>0 else (dataA,oversample(dataB,size_diff))

    def sub_sample_biggest(self,dataA:list,dataB:list):
        sub=lambda data:np.random.choice(data,min(len(dataA),len(dataB)),replace=False)
        return (sub(dataA),dataB) if len(dataA)>len(dataB) else (dataA,sub(dataB))

    def compute_per_sample_distance_from_saved(self,data_path: str,model: str,encoding_type:str,domain_A:str,domain_B:str,sample_strategy:str):
    
       model_domains=model.split('_')[0]
       model_iteration=model.split('_')[1]
       assert model_domains.split('2')[0]==domain_A
       assert model_domains.split('2')[1]==domain_B
       #PA_enc_ex_PA2MA_run_0_batch_392
       domain_A_data=glob(os.path.join(data_path,'{}_{}_{}_*.npy'.format(domain_A,encoding_type,model)))
       domain_B_data=glob(os.path.join(data_path,'{}_{}_{}_*.npy'.format(domain_B,encoding_type,model)))

       
    
       #function (list of path, list of path) --> list of path, list of path
       sample_domain_A_data,sample_domain_B_data=getattr(self,sample_strategy)(domain_A_data,domain_B_data)

       point_distances=[]
    
       for dp_AB in zip(sample_domain_A_data,sample_domain_B_data):
           dp_AB=(lambda A,B:(np.load(A),np.load(B)))(*dp_AB) # ([b,w,h,c] [b,w,h,c])

           point_distances.append(list(map(lambda A,B:self.compute(A,B),*dp_AB)))
       return np.concatenate(point_distances,axis=0)

    def compute_batched_distance_from_saved(self,data_path: str,model: str,encoding_type:str,domain_A:str,domain_B:str,sample_strategy:str,n_samples:int):
       model_domains=model.split('_')[0]
       model_iteration=model.split('_')[1]
       assert model_domains.split('2')[0]==domain_A
       assert model_domains.split('2')[1]==domain_B
       #PA_enc_ex_PA2MA_run_0_batch_392
       domain_A_data=glob(os.path.join(data_path,'{}_{}_{}_*.npy'.format(domain_A,encoding_type,model)))
       domain_B_data=glob(os.path.join(data_path,'{}_{}_{}_*.npy'.format(domain_B,encoding_type,model)))

       sample_domain_A_data,sample_domain_B_data=getattr(self,sample_strategy)(domain_A_data,domain_B_data)
    
       point_distances=[]
       for sample in range(len(sample_domain_A_data)//n_samples):
            AD_samples=[]
            BD_samples=[]
            for dp_AB in zip(sample_domain_A_data[sample:sample+n_samples],sample_domain_B_data[sample:sample+n_samples]):
                (lambda A,B:(AD_samples.append(np.load(A)),BD_samples.append(np.load(B))))(*dp_AB)
                #concat

            #compute
            BD_samples=np.concatenate(BD_samples,axis=0)
            AD_samples=np.concatenate(AD_samples,axis=0)

            point_distances.append(self.compute(AD_samples,BD_samples))
       return point_distances

# Maximum mean discrepancy

# Kernel functions

def radial_funtion(x,y):
    return np.linalg.norm(x-y)

def guassian_norm_funtion(x,y,sig=0.1):
    return np.exp((-(np.sum((x-y)**2,axis=(-3,-2,-1))))*(1/sig**2))

def normalized_guassian_norm_funtion(x,y,sig=0.1):
    return np.exp((-(np.sum((x-y)**2,axis=(-3,-2,-1))))*(1/(1e-7+np.std((x-y)**2,axis=(-3,-2,-1)))**2))

def scalar_cosine_similarity(x,y):
    x,y=x.reshape(1,-1),y.reshape(1,-1)
    return np.matmul(x,y.T)/((np.sum(x**2,axis=-1)**0.5)*(np.sum(y**2,axis=-1)**0.5))

cosine_similarity=np.vectorize(scalar_cosine_similarity,signature="(w,h,c),(w,h,c)->()")

# class
class MMD(distance_base):
    def __init__(self,kernel,kernel_trick=False):
        """
        kernel: function of x and y to scalar value
        kernel_trick: function with x,y input and xx,xy,yy output
        """
        self.kernel_trick=kernel_trick
        self.kernel=kernel
    def kernel_map(self,x,y):
        return self.kernel(np.stack(tuple(x for i in range(x.shape[0])),axis=1),y)
    def compute(self,x,y):
        """
        x: vector [Data points,data dimension]
        y: vector [Data points,data dimension]
        """
        if self.kernel_trick:
            xx,xy,yy=self.kernel_trick(x,y)
        else:
            #xx=np.mean(np.stack(tuple(x)))
            xx=np.mean(self.kernel_map(x,x))
            xy=np.mean(self.kernel_map(x,y))
            yy=np.mean(self.kernel_map(y,y))
        
        return np.mean(xx-2*xy+yy)


# Distribution Embedding
#[11.631524 17.959887 20.015646 20.278315 20.291876 20.295765 20.063093 19.747066 19.1053   18.75892  19.797281 20.028425 20.014605 19.846888 19.342031 19.854492 20.340357 19.967432 19.983932 20.460997 20.600676 20.471638 20.293932 20.012331 19.925253 20.451347 20.271711 20.173328 20.288008 20.328943 17.69984   9.143405]
#max 20.06 21
#min 9

def general_normal_distribution(X):
    """
    X [B,W,H,C]--> [B]
    """
    return np.mean(X),np.std(X)

def batch_normal_reduction(X):
    """
    X [B,W,H,C]--> [W,H,C]
    """
    return np.mean(X,axis=0),np.std(X,axis=0)

def multinomial_dist(X,range=(-6,6),bins=120):
    mu,sig=batch_normal_reduction(X)
    mu=rearrange(mu,'w h c -> c (w h)')
    sig=rearrange(sig,'w h c -> c (w h)')
    
    mu=(mu-np.mean(mu,axis=0))/np.std(mu,axis=0)
    sig=(sig-np.mean(sig,axis=0))/np.std(sig,axis=0)

    #mu,_=np.histogram(mu,range=range,bins=bins)
    mu=np.array(list(map(lambda mu:np.histogram(mu,range=range,bins=bins)[0] ,mu.tolist())))
    #sig,_=np.histogram(sig,range=range,bins=bins)
    sig=np.array(list(map(lambda mu:np.histogram(mu,range=range,bins=bins)[0] ,sig.tolist())))
    return mu,sig


# KL specific distances


def Entropy_of_Multinomial_gaussians_naive(P1,P2):
    mu_1=P1[0]+1e-9
    mu_2=P2[0]+1e-9
    sig_1=P1[1]**2+1e-9
    sig_2=P2[1]**2+1e-9
#def Entropy_of_Multinomial_gaussians_naive(mu_1,sig_1,mu_2,sig_2):
    H_mu=np.sum(entropy(mu_1,mu_2))
    H_sig=np.sum(entropy(sig_1,sig_2))
    return H_mu+H_sig

def Entropy_of_Multinomial_gaussians(P1,P2):
    mu_1=P1[0]
    mu_2=P2[0]
    sig_1=P1[1]**2+1e-9
    sig_2=P2[1]**2+1e-9
    M_H=np.array(
        list(
            map(
                lambda mu_1,sig_1,mu_2,sig_2:Univariate_Normal((mu_1,sig_1),(mu_2,sig_2)),
                mu_1,sig_1,mu_2,sig_2)
            )
        )
    return np.mean(M_H)

def Univariate_Normal(P1,P2):
    mu1=P1[0]
    mu2=P2[0]
    sig1=P1[1]**2
    sig2=P2[1]**2
    return 0.5*((mu1-mu2)**2/sig2+sig1/sig2-np.log(sig1/sig2)-1)

def moment_reduction(I,i,j):
    H,W,C=I.shape
    x_w=repeat(np.arange(H),'a -> a b c',b=W,c=C)**i
    y_w=repeat(np.arange(W),'a -> b a c',b=H,c=C)**j
    return np.sum(I*x_w*y_w)


def Moment_based_Multivariate_Normal(P1,P2):
    #mu_D1,sig_D1=batch_normal_reduction(P1)
    #mu_D1=moment_reduction(mu_D1)
    #sig_D1=moment_reduction(sig_D1)
    #mu_D2,sig_D2=batch_normal_reduction(P2)
    #mu_D2=moment_reduction(mu_D2)
    #sig_D2=moment_reduction(sig_D2)
    #KLD=-0.5*torch.mean(1+torch.log(z_sig.pow(2))-z_mean.pow(2)-z_sig.pow(2))
    mu1=P1[0]
    mu2=P2[0]
    sig1=P1[1]**2
    sig2=P2[1]**2
    return np.mean(0.5*((mu1-mu2)**2/sig2+sig1/sig2-np.log(sig1/sig2)-1))

#  KL-Divergence

class KL_divergence(distance_base):
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

class Jensen_Shanon_divergence(distance_base):
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

class Contrastive_Divergence(distance_base):
    def __init__(self,Model):
        """
        # REF
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