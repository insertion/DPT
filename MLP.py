
# coding: utf-8

# In[ ]:

# Multilayer Perceptron And Neural Network


# In[24]:

import theano,timeit,numpy
import dill as pickle
import theano.tensor as T
from LogisticRegression import load_data,LogisticReg
class HiddenLayer(object):
    def __init__(self,rng,input,n_in,n_out,W=None,b=None,activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh,not sigmod

        Hidden unit activation is given by: tanh(dot(input,W) + b)
        The more complicated the input distribution is, the more capacity the network will require to model it
        """
        self.input = input
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low  = -numpy.sqrt(6.0/(n_in+n_out)),
                    high = numpy.sqrt(6.0/(n_in+n_out)),
                    size=(n_in,n_out)),
                dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values,name='W',borrow=True)
        if b is None:
            b_value = numpy.zeros((n_out,),dtype=theano.config.floatX)
            b =theano.shared(value=b_value,borrow=True,name='b')
        self.W = W
        self.b = b
        
        line_output = T.dot(input,self.W) + self.b
        self.output = ( lin_output if activation is None else activation(line_output))
        self.params = [ self.W,self.b]
        # 隐藏层的参数
        
        


# In[163]:

class MLP(object):
    """
    Multi-Layer Perceptron Class
    
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """
    def __init__(self,rng,input,n_in,n_hidden,n_out):
        
        self.hiddenLayer = HiddenLayer(
            rng = rng,
            input = input,
            n_in = n_in,
            n_out = n_hidden,
            activation = T.tanh
        )
        
        self.topLayer = LogisticReg(
            input = self.hiddenLayer.output,
            n_in = n_hidden,
            n_out = n_out
        )
        # L1 and L2 regularization
        # by penalizing large values of the parameters, which decreases the amount of nonlinearity that the network models
        """
        L1  = T.sum(abs(param))
        L2  = T.sum(param ** 2)
        # the loss
        loss = NLL + lambda_1 * L1 + lambda_2 * L2
        """
        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.topLayer.W).sum()
        self.L2 = (self.hiddenLayer.W **2).sum() + (self.topLayer.W **2).sum()
        
        self.negative_log_likelihood = self.topLayer.negative_log_likelihood
        self.errors = self.topLayer.errors
        self.params = self.hiddenLayer.params + self.topLayer.params
        self.input = input
        """
        a = [1 ,2 ,4]
        b = [2 ,5, 7]
        a + b ==> [1, 2, 4, 2, 5, 7]
        """
        
        #cost = classifier.negative_log_likeihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2
        
        # given two lists of the same length, 
        # A = [a1, a2, a3, a4] 
        # B = [b1, b2, b3, b4]
        # C = zip(A,B)
        # zip generates a list C of same size, where each element is a pair formed from the two lists :
        # C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    
    
    
    
    


# In[21]:

def train_mlp(learning_rate = 0.01 ,L1_reg =0.0 ,L2_reg = 0.0001 ,n_epochs = 1000,
              dataset = 'mnist.pkl.gz',batch_size =20 ,n_hidden =500,n_out=10):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron
    """
    datasets = load_data(dataset)
    # import load_data from LogisticRegrecession
    
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    
    index = T.iscalar('index')
    x = T.matrix('x') # the data is presented as rasterized image 光栅化的图像
    y = T.ivector('y') # 1D vector label
    
    rng = numpy.random.RandomState(1234)
    #  1234 Random seed used to initialize the pseudo-random number generator
    
    classifier = MLP(rng= rng ,input= x,n_in =28*28,n_hidden = n_hidden,n_out = n_out)
    cost = classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg*classifier.L2
    
      
    ##################
    # UPDATE WEIGHTS #
    ##################
    gparams = [T.grad(cost=cost,wrt=param) for param in classifier.params]
    """
    # hat automatic differentiation (including theano's grad function) just uses the chain rule. 
    # The interesting point is that this is also how backpropagation works; it's just the chain rule.
    # under the hood, theano is using reverse-mode automatic differentiation which, 
    # in the case of neural nets, is equivalent to the backprop equations
    """
    # 多层网络的关键在于隐藏层的参数更新
    # bp算法，向后传递的是cost，然后从底层向前依次计算grad，前一层的输出作为后一层的输入
    # input，cost，cost=f(input) 都已经知道，可以求出梯度,cost不等于output
    # 把cost当成是params的函数，求cost的极小值
    updates = [(param ,param -learning_rate *gparam) for param,gparam in zip(classifier.params,gparams)] 
    
    update_model = theano.function(
        inputs = [index],
        outputs = cost,
        updates = updates,
        givens = {
            x:train_set_x[index * batch_size :(index +1) *batch_size],
            y:train_set_y[index * batch_size :(index +1) *batch_size]
        }
    )
    
    ##################
    # test_model     #
    # validate_model #
    ##################
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    
    message = ( 
                '############\n'                
                '# Training #\n'            
                '############\n'
               )
    print (message)
    
    patience = 10000
    patience_incease = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches,patience/2)
    
    best_validation_loss = numpy.inf
    epoch = 0
    stop_looping = False
    start_time = timeit.default_timer()
    while epoch < n_epochs and not stop_looping:
        epoch += 1
        for minibatch_index in range(n_train_batches):
            update_model(minibatch_index)
            iter =(epoch -1) * n_train_batches + minibatch_index
            if (iter  + 1) % validation_frequency ==0:
                validation_loss = numpy.mean([validate_model(i) for i in range(n_valid_batches)])
                if validation_loss < best_validation_loss:
                    
                    if validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience,iter*patience_incease)
                    
                    best_validation_loss = validation_loss
                    
                print 'epoch %i ,iteration %i,validation error %f %%' %(epoch,iter,validation_loss*100)
                
            if iter >= patience:
                stop_looping = True
                break
    
    with open('best_MLP_model.pkl', 'wb') as f:
        # this will overwrite current contents
        # 这里的dump相当与write
        pickle.dump(classifier, f)
        # 我们把变量从内存中变成可存储或传输的过程称之为序列化，
        # 在Python中叫pickling，在其他语言中也被称之为serialization，marshalling，flattening等等，都是一个意思
        # class method as a function


        # 当一个函数(function)定义在了class语句的块中（或者由 type 来创建的), 它会转成一个 unbound method ,
        # 当我们通过一个类的实例来 访问这个函数的时候，它就转成了 bound method , bound method 会自动把这个实例作为函数的地一个参数。
        # 实例方法即绑定方法不能被pickle，需要注册 使用copy_reg 或者直接使用dill代替pickle
        # 所以， bound method 就是绑定了一个实例的方法， 否则叫做 unbound method .它们都是方法(method), 是出现在 class 中的函数。
    end_time = timeit.default_timer()            
    print ( 'Optimization complete with best validation loss of %f %%' %(best_validation_loss*100)) 
    print 'The code run for %d epochs %d secs ,with %f epcohs/sec' %(epoch,end_time - start_time ,epoch*1.0/(end_time-start_time))
    
    
    
    
    
    
    
    


# In[ ]:

if __name__ == '__main__':
    train_mlp()

