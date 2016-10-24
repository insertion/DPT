# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 18:10:53 2016

@author: wlei
"""
"""
when you code in IDE,do not set theano in gpu mode ,as the IDE will freeze for along time
"""
import theano
import theano.tensor as T
from  theano.tensor.shared_randomstreams import RandomStreams
import numpy,timeit,os
from LogisticRegression import load_data

try:
    import PIL.Image as Image
except ImportError:
    import Image

from utils import tile_raster_images

"""
    It means that the representation is exploiting statistical regularities present in the training set,
    rather than merely learning to replicate the input.
    
    The auto-encode is lile PCA,if the hidden layer is non-linear, the auto-encoder behaves differently from PCA,
    with the ability to capture multi-modal aspects of the input distribution.
    
    普通的auto-encode
    It gives low reconstruction error on test examples from the same distribution as the training examples,
    but generally high reconstruction error on samples randomly chosen from the input space.
    
    the learning_rate is try differernt value ,then find the best one
"""

class DA(object):
    
    def __init__(self,numpy_rng,theano_rng=None,input=None,n_visible=28*28,n_hidden=500,W=None,b_hidden=None,b_visible=None):
        """
        Initialize the dA class by specifying the number of visible units (the
        dimension d of the input ),and the corruption level.
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        # create a Theano random generator that gives symbolic random values,
        # theano.rng generator symbolic value,
        # while numpy.rng generator real value(constant value)
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
            
        if not W:
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low = - 4 * numpy.sqrt(6.0/(n_hidden + n_visible)),
                    high = 4 * numpy.sqrt(6.0/(n_hidden + n_visible)),
                    size = (n_visible,n_hidden)
                ),
            )
            W = theano.shared(value = initial_W,name = 'w',borrow =True)
        
        if not b_hidden:
            b_hidden = theano.shared(
                value = numpy.zeros(
                    n_hidden,
                    dtype = theano.config.floatX
                ),
                name = 'b',
                borrow = True
            )
        if not b_visible:
            b_visible = theano.shared(
                value = numpy.zeros(
                    n_visible,
                    dtype = theano.config.floatX
                ),
                borrow = True
            )
        self.W = W
        self.b = b_hidden
        self.b_prime =b_visible
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        if  input:
            self.input = input
        else:
            self.input = T.dmatrix(name = 'input')
        self.params = [self.W,self.b,self.b_prime]
        
    def get_corrupted_input(self,input,corruption_level):
        
        return self.theano_rng.binomial(size = input.shape,n=1,p=1-corruption_level,dtype = theano.config.floatX) * input
        
        """
        随机值服从二项分布(伯努利分布)，n是伯努利试验的次数，p是概率，当 n= 1 时变成(0,1)分布，结果不是1就是0
        1 has a probability of 1 - corruption_level and 0 with ``corruption_level``
        这样就随机的把input的某一部分擦除掉 (设为 0)
        """
    def get_hidden_layer(self,input):
        """
            compute the hiiden layer units
        """
        return T.nnet.sigmoid(T.dot(input,self.W)+self.b)
        
    def get_reconstructed_input(self,hidden_layer):
        
        return T.nnet.sigmoid(T.dot(hidden_layer,self.W_prime)+self.b_prime)
    
    def get_cost_updates(self,corruption_level,learning_rate):
        
        x = self.get_corrupted_input(self.input,corruption_level)
        # corrupted input
        y = self.get_hidden_layer(x)
        # code
        z = self.get_reconstructed_input(y)
        
        cross_entropy =  -T.sum(self.input * T.log(z) + (1-self.input)*T.log(1-z),axis =1 )
        # 这里把input当成真实概率分布,z作为预测概率分布
        # 我们的优化目标是使交叉熵尽可能小
        cost = T.mean(cross_entropy)
        
        grads = T.grad(cost,self.params)
        
        updates =[
            (parm , parm - learning_rate*grad) for parm,grad in zip(self.params,grads)
        ]
        return (cost,updates)
        
def learn_DA(learning_rate=0.1,epochs=15,dataset='mnist.pkl.gz',batch_size=20,output_folder='DA_plots',corruption_level=0.0):
    '''
         当corruption_level ==0的时候，普通的自编码
    '''
    
    datasets = load_data(dataset)  
    train_set_x, train_set_y = datasets[0]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images  
      
    ######################
    # BUILDING THE MODEL #
    ######################  
    
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    
    # create the instance of class DA
    
    da = DA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible = 28*28,
        n_hidden=500
    )
    
    (cost,updates) = da.get_cost_updates(corruption_level=corruption_level,learning_rate=learning_rate)

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size : (index + 1) * batch_size]
        },
        on_unused_input='ignore',
    )
    start_time = timeit.default_timer()
    
    ############
    # TRAINING #
    ############
    
    for epoch in range(epochs):
        for batch_index in range(n_train_batches):
            train_da(batch_index)
           # print ('epoch %i,iter %i'%(epoch,epoch* n_train_batches +batch_index))
    end_time = timeit.default_timer()
    
    training_time = (end_time - start_time)

    print(('The %i%% corruption code for file '%(corruption_level *100) +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % (training_time / 60.)))
    # os.path.split(__file__) 把文件和路径分割成一个tuple(foldername,filename)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)       
    # In order to get a feeling of what the network learned we are going to plot the filters (defined by the weight matrix)
    image = Image.fromarray(
        tile_raster_images(
            X = da.W.get_value(borrow=True).T,
            img_shape = (28,28),
            tile_shape = (10,10),
            tile_spacing = (1,1)
        )
    )
    # This function is useful for visualizing datasets whose rows are images
    image.save('filters_corruption_%i.png'%(corruption_level*100))
    os.chdir('../')

if __name__ == '__main__':
    for i in range(10):
        learn_DA(corruption_level=i*1.0/10)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        