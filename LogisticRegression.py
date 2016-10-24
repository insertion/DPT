
# coding: utf-8

# # Classifying MNIST digits using Logistic Regression

#   **shared variable** : 相当于c语言中的全局变量，GPU会在shared variable被创建时把它复制到gpu中，而普通的变量相当于局部变量，函数执行时才赋值

# In[13]:

import theano
import theano.tensor as T
import numpy,gzip,timeit
import cPickle as pickle
class LogisticReg(object):
    def __init__(self,input,n_in,n_out):
        self.W =theano.shared(value=numpy.zeros((n_in,n_out),dtype=theano.config.floatX),name='W',borrow=True)
        self.b =theano.shared(value=numpy.zeros((n_out,),dtype=theano.config.floatX),name='b',borrow=True)
        # 参数W和b初始化为0
        # input 600*（28*28）600个样本 
        # W（28*28）*10  每列表示一种类别的参数
        # b 1*10
        self.p_y_given_x =T.nnet.softmax(T.dot(input,self.W)+self.b)
        '''
            import theano,numpy
            import theano.tensor as T
            y = T.nnet.softmax(numpy.array([1.2,3.3,4.3]) )
            out = theano.function([],y,on_unused_input='ignore')
            print out()
            或者如下:
            # 打印tensor变量
            y = T.nnet.softmax( numpy.array([1.2,3.3,4.3]) )
            print y.eval()
            eval的第一个参数是一个dict，表示input，key是变量名，value是变量的值，其背后也是构建function
        '''
        # theano的变量是无法直接输出，只能通过函数输出，tensor变量只是占位符，真正的运算要编译function后才执行
        # softmax的输入是向量或者矩阵，不改变维度
        # The softmax function will, when applied to a matrix, compute the softmax values row-wise.
        # 600*10矩阵 每个样本是各种类别中的概率
        self.y_pred =T.argmax(self.p_y_given_x,axis=1)
        # 600*1 每个样本预测的类别
        # axis = 0 表示逐列处理 axis = 1 表示逐行处理
        # max return the vluae of max along a give axis
        # Returns the index of the maximum value along a given axis
        # y = arg max f(t) return f(t)函式中产生最大output的那个参数 T
        self.params =[self.W,self.b]
        self.input =input
    
    def negative_log_likelihood(self,y):
        # y=600*1 shape[0]=600
        # T.arange(y.shape[0]) is a symbolic vector which will be [0,1,2,... n-1]
        # 最大似然 minibatch的联合概率
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])
        # 用整数序列来对array读取
        # mean allows for the learning rate choice to be less dependent of the minibatch size.
    
    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # ndim向量的维度
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
            # T.neq(x,y) 返回一个向量，x!=y 返回 1 
            # 返回错误率
        else:
            raise NotImplementedError()
    


# In[14]:

def load_data(dataset):
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)  
            # tuple(input, target)
    #函数嵌套
    def shared_data(data_xy,borrow=True):
        data_x,data_y = data_xy
        #把二维数据分解成 input 和 target
        shared_x = theano.shared(numpy.asarray(data_x,dtype=theano.config.floatX),borrow=True)
        shared_y = theano.shared(numpy.asarray(data_y,dtype=theano.config.floatX),borrow=True)
        # numpy.asarray(a, dtype=None, order=None)  Convert the input a to an array.
        return shared_x ,T.cast(shared_y,'int32')

    test_set_x,test_set_y   = shared_data(test_set)
    valid_set_x,valid_set_y = shared_data(valid_set)
    train_set_x,train_set_y = shared_data(train_set)
    return [(train_set_x,train_set_y),(valid_set_x,valid_set_y),(test_set_x,test_set_y)]


# In[15]:

def sgd_opt(learning_rate=0.12,n_epochs=1000,dataset='mnist.pkl.gz',batch_size=600):     
    # 参数优化
    datasets = load_data(dataset)
    train_set_x,train_set_y = datasets[0]
    valid_set_x,valid_set_y = datasets[1]
    test_set_x,test_set_y   = datasets[2]
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size 
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches  = test_set_x.get_value(borrow=True).shape[0] / batch_size

    index = T.lscalar()
    x = T.matrix('data')#28*28
    y = T.ivector('labels')#presented as 1D vector of [int] labels
    # x and y take part in 
    # x is an implicit symbolic input to the definition of cost, 
    # because the symbolic variables of classifier were defined in terms of x at initialization
    classifier = LogisticReg(input=x,n_in=28*28,n_out=10)
    cost = classifier.negative_log_likelihood(y)

    g_W = T.grad(cost=cost,wrt=classifier.W)
    g_b = T.grad(cost=cost,wrt=classifier.b)
    # auto get the gradient
    updates=[ (classifier.W,classifier.W-learning_rate * g_W),
             (classifier.b,classifier.b-learning_rate * g_b) ]

    # updates = { classifier.W:classifier.W-learning_rate * g_W , classifier.b:classifier.b-learning_rate * g_b }
    # The parameter 'updates' of theano.function() expects an OrderedDict, got <type 'dict'>.
    # or use a list of (shared, update) pairs
    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`

    train_model = theano.function( 
            [index],
            cost,
            updates=updates,
            givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]} )
    # The entire learning algorithm thus consists in looping over all examples in the datase
    # 返回值是cost
    # 每次调用train_model 都会更新参数W和b，得到新的cost
    # index 是调用train_model这个函数时输入的
    # on every function call, it will first replace x and y 
    # with the slices from the training set specified by index.

   
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
            }
    )
    
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    ###############
    # TRAIN MODEL #
    ###############
    start_time = timeit.default_timer()
    
    epoch=0
    stop_looping=False
    patience = 5000 #至少迭代5000次
    patience_increase =2 #迭代次数的指数
    improvement_threshold=0.995 #每次至少有0.005的性能提升，迭代次数才会增加
    validation_frequency = min(n_train_batches ,patience/2 )
    best_validation_loss = numpy.inf#无穷大
    
    while epoch < n_epochs and not stop_looping:
        epoch = epoch + 1
        #每个epoch遍历一遍样本
        for minibatch_index in range(n_train_batches):
            train_model(minibatch_index)
            #interation number
            iterNum = (epoch - 1) * n_train_batches + minibatch_index
            if iterNum % validation_frequency ==0:
                validation_loss = numpy.mean([validate_model(i) for i in range(n_valid_batches)])
                print 'epoch %i,validation error %f %%' %(epoch,validation_loss*100)
    ##################
    # Early-stopping #
    ##################
    # Early-stopping combats overfitting by monitoring the model’s performance on a validation set
    # If the model’s performance ceases to improve sufficiently on the validation set, or even degrades with further optimization, 
    # then the heuristic implemented here gives up on much further optimization.
                if validation_loss < best_validation_loss:
                    if validation_loss < best_validation_loss *  improvement_threshold:
                        patience =  max(patience,iterNum * patience_increase)
                    best_validation_loss = validation_loss
                if iterNum > patience:
                    stop_looping = True
                    break
    
    # save the best model
    with open('best_LR_model.pkl', 'wb') as f:
        # this will overwrite current contents
        # 这里的dump相当与write
        pickle.dump(classifier, f)
    end_time = timeit.default_timer()            
    print ( 'Optimization complete with best validation score of %f %%' %(best_validation_loss*100)) 
    print 'The code run for %d epochs %d secs ,with %f epcohs/sec' %(epoch,end_time - start_time ,epoch*1.0/(end_time-start_time))
    
    
 


# In[17]:

def predict():
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    # load the saved model
    classifier = pickle.load(open('best_model.pkl'))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)
    # 模型已经训练好了，把test_set输入分类器的input中，y_pred就是预测的结果
    # We can test it on some examples from test test
    dataset='mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[:10])
    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values)


if __name__ == '__main__':
    sgd_opt()

