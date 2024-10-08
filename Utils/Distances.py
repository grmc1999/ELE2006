
import numpy as np

# Maximum mean discrepancy

# Kernel functions

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
        if kernel_trick:
            xx,xy,yy=kernel_trick(x,y)
        else:
            xx=np.array(list(map(lambda x,y: self.kernel(x,y),x,x)))
            xy=np.array(list(map(lambda x,y: self.kernel(x,y),x,y)))
            yy=np.array(list(map(lambda x,y: self.kernel(x,y),x,x)))
        
        return np.mean(xx-2*xy+yy)

# KL-Divergence


# Jensen-Shanon Divergence


# Contrastive Divergence

# Score Matching

# Noise Contrastive Estimation

# Probability flow