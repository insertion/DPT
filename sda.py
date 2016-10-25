# -*- coding: utf-8 -*-

"""
    rule:
        comment after the code!

    Note that after pretraining, the SdA is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.

    when you use a numpy ndarry or a Python number together 
    with TensorVariable instances in arithmetic expressions, 
    the result is a TensorVariable. What happens to the ndarray or the number? 
    Theano requires that the inputs to all expressions be Variable instances, 
    so Theano automatically wraps them in a TensorConstant.
"""

import numpy,timeit,os
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
"""
    Because in Theano you first express everything symbolically 
    and afterwards compile this expression to get functions, 
    using pseudo-random numbers is not as straightforward as it is in NumPy
"""
from LogisticRegression import LogisticReg,load_data
from MLP import HiddenLayer
from Denoising_autoencode import DA

class SdA(object):
    """
        contruct the architexture
    """
    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        n_ins=28*28,
        hidden_layers_sizes=[500,500],
        n_outs=10,
        corruption_levels=[0.1,0.1]
    ):
        self.sigmoid_layers = []
        self.DA_layers = []
        self.params =[]
        self.n_layers = len(hidden_layers_sizes)
        # n_layers is the depth of our model
        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2**30))
        

        self.x = T.matrix('imput')
        self.y = T.ivector('labels')

        for i in range(self.n_layers):
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i-1]
        # the size of the input is either the number of hidden units of
        # the layer below or the input size if we are on the first layer


            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output
                # theano的赋值是图结构,变量具有别名特性，占位符，layer_input 与sigmod.output相连接
                # sigmod.output更新后，layer_input 也会更新，layer-wise 更新
            sigmoid_layer = HiddenLayer(
                        rng = numpy_rng,
                        input = layer_input,
                        n_in = input_size,
                        n_out = hidden_layers_sizes[i],
                        activation = T.nnet.sigmoid             
            )
            self.sigmoid_layers.append(sigmoid_layer)

            self.params.extend(sigmoid_layer.params)
            # append add one element,extend add a list of element

            DA_layer = DA(
                numpy_rng = numpy_rng,
                input = layer_input,
                n_visible = input_size,
                n_hidden = hidden_layers_sizes[i],
                W = sigmoid_layer.W,
                b_hidden = sigmoid_layer.b 
            )
            self.DA_layers.append(DA_layer)
            # share paramseter with sigmoid_layer
        
        self.topLayer = LogisticReg(
            input = self.sigmoid_layers[-1].output,
            n_in = hidden_layers_sizes[-1],
            n_out = n_outs
        )

        self.params.extend(self.topLayer.params)
        #用新列表扩展原来的列表
        self.finetune_cost = self.topLayer.negative_log_likelihood(self.y)

        self.errors = self.topLayer.errors(self.y)


    def pretraining_functions(self,train_set_x,batch_size):
        """Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes."""

        index = T.lscalar('index')
        corruption_level = T.scalar('corruption')
        learning_rate = T.scalar('rate')
        # theano.function 的参数，参数名是'rate' not 'learning_rate'
        # learning_rate 是指向值的占位符，不是name,name是括号里的
        batch_begin = index * batch_size
        batch_end   = index * batch_size + batch_size

        pretrain_fns = []
        for DA in self.DA_layers:

            cost,updates = DA.get_cost_updates (corruption_level,learning_rate)

            fn = theano.function(
                inputs = [
                    index,
                    theano.In(corruption_level,value=0.2),
                    theano.In(learning_rate,value = 0.1) 
                ],
                outputs = cost,
                updates = updates,
                givens = {
                    self.x :train_set_x[batch_begin:batch_end]
                }
            )

            pretrain_fns.append(fn)
        return pretrain_fns
    def build_finetune_functions(self,datasets,batch_size,learning_rate):

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y)   = datasets[2]

        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
        n_test_batches  = test_set_x.get_value(borrow=True).shape[0] / batch_size

        index = T.lscalar('index')
        grads = T.grad(self.finetune_cost,self.params)
        updates = [
            (param,param-grad*learning_rate)
            for param,grad in zip(self.params,grads)
        ]


        train_fn = theano.function(
            inputs = [index],
            outputs = self.finetune_cost,
            updates = updates,
            givens = {
                self.x : train_set_x[index * batch_size:(index +1) *batch_size],
                self.y : train_set_y[index * batch_size:(index +1) *batch_size]
            },
            name = 'train'
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens ={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name = 'test'
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )
         # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_score, test_score


# Demonstrates how to train and test a stochastic denoising autoencoder.
def test_sda(
    finetune_lr=0.1,
    pretraining_epochs=15,
    pretrain_lr =0.001,
    training_epochs =1000,
    dataset = 'mnist.pkl.gz',
    batch_size=1
):
    """
    batch_size =1 cause SGD not batch gd
    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    # if Theano is using a GPU device, then the borrow flag has no effect
    # When borrow=True is passed to get_value, 
    # it means that the return value might be aliased to some of Theano’s internal memory

    numpy_rng = numpy.random.RandomState(seed = 1234)
    print('... building the SdA model')
    sda = SdA(
        numpy_rng = numpy_rng,
        n_ins = 28*28,
        hidden_layers_sizes = [1000,1000,1000],
        n_outs = 10
    )
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print('... getting the pretraining functions')
    pretraining_fns = sda.pretraining_functions(
        train_set_x = train_set_x,
        batch_size  = batch_size
    )
    print('... pre-training the model')

    start_time = timeit.default_timer()

    corruption_levels = [0.1,0.2,0.3]
    for i in range(sda.n_layers):
        """
        Pre-train layer-wise
        """
        for epoch in range(pretraining_epochs):
            for batch_index in range(n_train_batches):
                pretraining_fns[i](
                    index = batch_index,
                    corruption = corruption_levels[i],
                    rate = pretrain_lr
                )
    end_time = timeit.default_timer()
    print(('The pretraining code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))

    ########################
    # FINETUNING THE MODEL #
    ########################
    print('... getting the finetuning functions')

    train_fn,validation_model,test_model  = sda.build_finetune_functions(
        datasets = datasets,
        batch_size = batch_size,
        learning_rate = finetune_lr
    )

    print('... finetunning the model')

    patience = 2 * n_train_batches
    patience_increse = 2

    improvement_threshold = 0.995

    validation_frequency = min(n_train_batches,patience/2)

    best_validation_loss = numpy.inf
    test_score = 0
    start_time = timeit.default_timer()
    stop_looping = False
    epoch = 0

    while epoch < training_epochs and not stop_looping:
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            train_fn(minibatch_index)
            iter  = (epoch -1 ) * n_train_batches + minibatch_index

            if (iter + 1)%validation_frequency ==0:
                validation_loss = numpy.mean(validation_model())
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       validation_loss * 100.))
                if validation_loss < best_validation_loss:
                    if validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience,iter * patience_increse)
                        # 这里patience 增长太快了，指数增长
                    best_validation_loss = validation_loss
                    test_loss = numpy.mean(test_model())
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_loss * 100.))
            
            if patience <= iter:
                stop_looping = True
                break
    end_time = timeit.default_timer()
    print(('The training code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))
if __name__ == '__main__':
    test_sda()