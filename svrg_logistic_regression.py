from __future__ import print_function
import os

try:
    os.makedirs('./svrg_optimization')
    import requests as rq
    init_svrg = rq.get('https://raw.githubusercontent.com/apache/incubator-mxnet/master/python/mxnet/contrib/svrg_optimization/__init__.py')
    with open('./svrg_optimization/__init__.py', 'w') as file_init:
        file_init.write(init_svrg.text)

    modu_svrg = rq.get('https://raw.githubusercontent.com/apache/incubator-mxnet/master/python/mxnet/contrib/svrg_optimization/svrg_module.py')
    with open('./svrg_optimization/svrg_module.py', 'w') as file_modu:
        file_modu.write(modu_svrg.text)

    opti_svrg = rq.get('https://raw.githubusercontent.com/apache/incubator-mxnet/master/python/mxnet/contrib/svrg_optimization/svrg_optimizer.py')
    with open('./svrg_optimization/svrg_optimizer.py', 'w') as file_opti:
        file_opti.write(opti_svrg.text)
        
except FileExistsError:
    pass

from svrg_optimization.svrg_module import SVRGModule

# LOGISTIC REGRESSION SVRG
import mxnet as mx
svrg = False

# define hyperparameters
lr = 0.2
batch_size = 32
epochs = 100
step = 300
ctx = mx.gpu()
ctx = mx.cpu()

# define dataset and dataloader
mnist = mx.test_utils.get_mnist()

train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True,
                            data_name='data', label_name='softmax_label')
test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size * 2, shuffle=False,
                            data_name='data', label_name='softmax_label')

if svrg:

    svrg_logistic(train_iter, test_iter)
    

def svrg_logistic(train_iter, test_iter):
    # define symbol
    data = mx.sym.var('data')  # (bs, 1, 28, 28)
    label = mx.sym.var('softmax_label')
    data = mx.sym.Flatten(data)  # (bs, 28*28)
    fc = mx.sym.FullyConnected(data, num_hidden=10, name='fc')
    logist = mx.sym.SoftmaxOutput(fc, label=label, name='softmax')

    metric_list = [mx.metric.Accuracy(output_names=['softmax_output'], label_names=['softmax_label']),
                mx.metric.CrossEntropy(output_names=['softmax_output'], label_names=['softmax_label'])]
    eval_metrics = mx.metric.CompositeEvalMetric(metric_list)

    mod = mx.mod.Module(logist, data_names=['data'], label_names=['softmax_label'], context=ctx)
    optimizer_params = {'learning_rate': lr} # optimizer_params={'lr_scheduler': mx.lr_scheduler.FactorScheduler(step=10, factor=0.99)}
    svrg_model = SVRGModule(symbol=logist, data_names=['data'], label_names=['softmax_label'], context=ctx, update_freq=1)
    svrg_model.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    svrg_model.init_params()#(initializer=mx.init.Zero(), allow_missing=False, force_init=False, allow_extra=False)
    svrg_model.init_optimizer(kvstore='local', optimizer='sgd', optimizer_params=optimizer_params)
    import logging
    logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
    # create a trainable module on compute context
    #mlp_model = mx.mod.Module(symbol=mlp, context=ctx)
    svrg_model.fit(train_iter,  # train data
                eval_data=test_iter,  # validation data
                #optimizer='sgd',  # use SGD to train
                #optimizer_params = (('learning_rate', lr),),  # use fixed learning rate
                eval_metric=eval_metrics,  # report accuracy during training
                batch_end_callback = mx.callback.Speedometer(batch_size, 500), # output progress for each 100 data batches
                epoch_end_callback=mx.callback.do_checkpoint('logistic', 10),
                num_epoch=epochs)  # train for at most 10 dataset passes
    val_score = svrg_model.score(val_iter, 'acc')[0][1]
    train_score = svrg_model.score(train_iter, 'acc')[0][1]
    print(val_score, train_score)

    #TODO: Define function returns
    lr_params_svrg.update({lr: [val_score, train_score]})