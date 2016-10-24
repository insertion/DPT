# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 17:17:46 2016

@author: wlei
"""



import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
import dill as pickle
import timeit,numpy
from LogisticRegression import load_data,LogisticReg
from  theano.tensor.shared_randomstreams import RandomStreams
from MLP import HiddenLayer

'''
theano 的 function 有的参数需要是 tensor variable 有的需要 constant variable
'''
class cnnClassifier(object):
    def __init__(self):
        """
        actual model which can be dump and load
        """
        pass

class LeNetConvPoolLayer(object):
    """
    put conv layer and pool layer togather
    The lower-layers are composed to alternating convolution and max-pooling layers
    """
    def __init__(self,rng,input,filter_shape,image_shape,poolsize=(2,2)):
        """
        :param filter_shape: (number of filters, #卷积层的深度,output channels
                              num input feature maps,#输入层的深度 input channels
                              filter height, 
                              filter width
                             )
                              
        :param image_shape: (batch size, 
                             num input feature maps,
                             image height, 
                             image width
                            )
        """
        
        assert image_shape[1] == filter_shape[1]
        #卷积核的depth要和输入的depth一样
        #assert的参数是一个布尔表达式，如果该表达式值为0，则程序被迫退出，且在终端输出相关信息
        #number of input feature maps is channels(r,g,b),is also called depth
        
        # there are "num input feature maps (depth) * filter height * filter width" inputs to each hidden unit
        fn_in = numpy.prod(filter_shape[1:])
        fn_out = (filter_shape[0] * numpy.prod(filter_shape[2:])/numpy.prod(poolsize))
        #卷积核的输入输出
        
        W_bound = numpy.sqrt(6.0/(fn_in + fn_out))
        
        self.W = theano.shared(
            numpy.asarray(
                            rng.uniform(low = -W_bound,high = W_bound,size = filter_shape),
                            dtype = theano.config.floatX
                        ),
            borrow = True
        )
        
        
        b_values = numpy.zeros(filter_shape[0],dtype = theano.config.floatX)
        self.b = theano.shared(b_values,borrow=True)
        """
            These replicated units share the same parameterization (weight vector and bias) and form a feature map.
            bias's size equal to the number of filters
        """
        conv_out =    conv2d(
                            input=input,
                            #s ymbolic 4D tensor
                            filters=self.W,
                            filter_shape=filter_shape,
                            input_shape=image_shape
                            # input shape should be None or a tuple of constant int values
                            )
        # return : 4Dtensor (batch size, output channels, output rows, output columns)
        
        pool_out=pool.pool_2d(
                            input = conv_out,
                            ds=poolsize,
                            ignore_border = True
                            )
                            
        self.input = input
        self.output = T.tanh(pool_out + self.b.dimshuffle('x',0,'x','x'))
        # along a dimension is to traverse this dimension(from 0 to end)
        # dimshuffle(0, 'x', 1) -> AxB to Ax1xB
        # bais is set to output channels
        # activation function can be put in pool layer
        self.params = [self.W,self.b]

def trian_LeNet(learning_rate = 0.1,n_epochs=1000,dataset='mnist.pkl.gz',nkernels=[20,50],batch_size=500):
    """
        :param nkerns: number of kernels on each layer
        
    """
    rng = numpy.random.RandomState(1234)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    datasets = load_data(dataset)
    
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y   = datasets[2]
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches  = test_set_x.get_value(borrow=True).shape[0]  / batch_size
    
    index = T.lscalar()
    x = T.matrix('input')
    y = T.ivector('label')
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('[building  the model]')

    def get_corrupted_input(input,corruption_level):
        return theano_rng.binomial(size = input.shape,n=1,p=1-corruption_level,dtype = theano.config.floatX) * input
    
    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # class's params should be real value ,not tensor vaiable
    layer0_input =  get_corrupted_input(x.reshape((batch_size,1,28,28)),0.1)
    
    layer0 = LeNetConvPoolLayer(
                rng,
                input=layer0_input,
                image_shape=(batch_size,1,28,28),
                filter_shape = (nkernels[0], 1, 5, 5),
                poolsize=(2,2)
                )
    print('.....................layer0 is built,(conv+pool)')   

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)       
    layer1 = LeNetConvPoolLayer(
                rng,
                input= layer0.output,
                image_shape=(batch_size,nkernels[0],12,12),
                filter_shape=(nkernels[1],nkernels[0],5,5),
                poolsize=(2,2)
                )
    print('.....................layer1 is built,(conv+pool)')
    
    # the HiddenLayer being fully-connected, it operates on 2D matrices of shape (batch_size, num_pixels)
    layer2_input= layer1.output.flatten(2)
    #layer2_input = layer1.output.reshape((batch_size,numpy.prod(layer1.output[1:])))
        
    """
     if we flatten a tensor of shape (2, 3, 4, 5) with flatten(x, outdim=2),
     then we’ll have the same (2-1=1) leading dimensions (2,),
     and the remaining dimensions are collapsed. So the output in this example would have shape (2, 60)
    
    """
    layer2 = HiddenLayer(
                rng,
                input=layer2_input,
                n_in=nkernels[1] * 4 * 4,
                n_out = 500,
                activation=T.tanh
                )
    
    layer3 = LogisticReg(input= layer2.output,n_in=500,n_out=10)
    
    print('.....................layer2,layer3 are built,(MLP+LR)')
    print('[compiling the model]')
    classifier = layer3
    cost = classifier.negative_log_likelihood(y)
    # create a list of all model parameters to be fit by gradient descent,list add is just append one after one
    params = layer0.params + layer1.params + layer2.params + layer3.params
    grads = T.grad(cost,params)
    
    updates = [
        (param,param-learning_rate*grad) for param ,grad in zip(params,grads)
        ]
        
    update_model = theano.function(
        [index],
        [cost,classifier.errors(y)],
        # Outputs must be theano Variable or Out instances
        updates=updates,
        givens = {
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )
        
    validate_model = theano.function(
        [index],
        classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
        ) 
    
    test_model = theano.function(
        [index],
        classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
            }
        ) 
    def clean_params():
        #重新训练
        pass
    def traing():
        pass
    ################################################################################################################                    
    message = ( 
                '############################\n'                
                '# Compliled,start training #\n'            
                '############################\n'
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
            train_cost,train_error = update_model(minibatch_index)
            iter =(epoch -1) * n_train_batches + minibatch_index
            if (iter  + 1) % validation_frequency ==0:
                validation_loss = numpy.mean([validate_model(i) for i in range(n_valid_batches)])
                if validation_loss < best_validation_loss:
                    if validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience,iter*patience_incease)
                    best_validation_loss = validation_loss
                    
                print 'epoch %i ,iteration %i,training cost %f %%,train error %f %%,validation error %f %%' %(epoch,iter,train_cost*100,train_error*100,validation_loss*100)
            if iter >= patience:
                stop_looping = True
                break
    test_loss = numpy.mean([test_model(i) for i in range(n_test_batches)])
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
    
#def predict():
#    """
#    An example of how to load a trained model and use it
#    to predict labels.
#    """
#
#    # load the saved model
#    classifier = pickle.load(open('best_model.pkl'))
#
#    # compile a predictor function
#    predict_model = theano.function(
#        inputs=[classifier.input],
#        outputs=classifier.y_pred)
#    # 模型已经训练好了，把test_set输入分类器的input中，y_pred就是预测的结果
#    # We can test it on some examples from test test
#    dataset='mnist.pkl.gz'
#    datasets = load_data(dataset)
#    test_set_x, test_set_y = datasets[2]
#    test_set_x = test_set_x.get_value()
#
#    predicted_values = predict_model(test_set_x[:10])
#    print("Predicted values for the first 10 examples in test set:")
#    print(predicted_values)


if __name__ == '__main__':
    # 增加命令行参数获取
    trian_LeNet()
                   
                            
                            
                            
                            
                            
                            
                            
                            
        