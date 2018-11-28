
#%%
# -*- coding: utf-8 -*-
"""Hello, Colaboratory

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/notebooks/welcome.ipynb
"""
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


#%%
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

if svrg:

    # define dataset and dataloader
    mnist = mx.test_utils.get_mnist()

    train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True,
                                data_name='data', label_name='softmax_label')
    test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size * 2, shuffle=False,
                                data_name='data', label_name='softmax_label')

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
    lr_params_svrg.update({lr: [val_score, train_score]})


#%%
# LOGISTIC REGRESSION SGD

# define dataset and dataloader
mnist = mx.test_utils.get_mnist()

train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True,
                               data_name='data', label_name='softmax_label')
test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size * 2, shuffle=False,
                              data_name='data', label_name='softmax_label')

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
import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
mod.fit(
    train_data=train_iter,
    eval_data=test_iter,
    optimizer='sgd',
    optimizer_params={'learning_rate': lr},
    batch_end_callback=mx.callback.Speedometer(batch_size, 500),
    epoch_end_callback=mx.callback.do_checkpoint('logistic', 10),
    eval_metric=eval_metrics,
    num_epoch=epochs
)


#%%
# FULLY CONNECTED NETWORK

import mxnet as mx

mnist = mx.test_utils.get_mnist()

# Fix the seed
mx.random.seed(42)

# Set the compute context, GPU is available otherwise CPU
ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

batch_size = 100
train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

data = mx.sym.var('data')
# Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)
data = mx.sym.flatten(data=data)

# The first fully-connected layer and the corresponding activation function
fc1  = mx.sym.FullyConnected(data=data, num_hidden=64)
act1 = mx.sym.Activation(data=fc1, act_type="relu")

# The second fully-connected layer and the corresponding activation function
fc2  = mx.sym.FullyConnected(data=act1, num_hidden = 32)
act2 = mx.sym.Activation(data=fc2, act_type="relu")

# MNIST has 10 classes
fc3  = mx.sym.FullyConnected(data=act2, num_hidden=10)
# Softmax with cross entropy loss
mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')




#%%
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np

data_ctx = mx.cpu()
#model_ctx = mx.cpu()
model_ctx = mx.gpu()

batch_size = 64
num_inputs = 784
num_outputs = 10
num_examples = 60000
def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)
train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                              batch_size, shuffle=False)

net = gluon.nn.Dense(num_outputs)
net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1,784))
        label = label.as_in_context(model_ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

evaluate_accuracy(test_data, net)

epochs = 10
moving_loss = 0.

for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx).reshape((-1,784))
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)
        cumulative_loss += nd.sum(loss).asscalar()

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))


#%%
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np

data_ctx = mx.cpu()
#model_ctx = mx.cpu()
model_ctx = mx.gpu()


batch_size = 64
num_inputs = 784
num_outputs = 10
num_examples = 60000
def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)
train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                              batch_size, shuffle=False)

net = gluon.nn.Dense(num_outputs)
net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

optimizer_params = {'learning_rate': lr}#optimizer_params={'lr_scheduler': mx.lr_scheduler.FactorScheduler(step=10, factor=0.99)}
svrg_model = SVRGModule(symbol=net, context=model_ctx, update_freq=5)
#svrg_model.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
#svrg_model.init_params()#(initializer=mx.init.Zero(), allow_missing=False, force_init=False, allow_extra=False)


trainer = gluon.Trainer(net.collect_params(), svrg_model, {'learning_rate': 0.1})

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1,784))
        label = label.as_in_context(model_ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

evaluate_accuracy(test_data, net)

epochs = 10
moving_loss = 0.

for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx).reshape((-1,784))
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)
        cumulative_loss += nd.sum(loss).asscalar()

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))

net = gluon.nn.Dense(num_outputs)


#%%
lr_params_svrg = {}
#Create a SVRGModule

train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
for lr_ in list(range(100, 10, -20)):
    lr = lr_/100
    print(lr)
    optimizer_params = {'learning_rate': lr}#optimizer_params={'lr_scheduler': mx.lr_scheduler.FactorScheduler(step=10, factor=0.99)}
    svrg_model = SVRGModule(symbol=mlp, context=ctx, update_freq=5)
    svrg_model.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    svrg_model.init_params()#(initializer=mx.init.Zero(), allow_missing=False, force_init=False, allow_extra=False)
    #svrg_model.init_optimizer(kvstore='local', optimizer='sgd', optimizer_params=optimizer_params)
    import logging
    logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
    # create a trainable module on compute context
    #mlp_model = mx.mod.Module(symbol=mlp, context=ctx)
    svrg_model.fit(train_iter,  # train data
                  eval_data=val_iter,  # validation data
                  optimizer='sgd',  # use SGD to train
                  optimizer_params = (('learning_rate', lr),),  # use fixed learning rate
                  eval_metric='acc',  # report accuracy during training
                  batch_end_callback = mx.callback.Speedometer(batch_size, 1000), # output progress for each 100 data batches
                  num_epoch=10)  # train for at most 10 dataset passes
    val_score = svrg_model.score(val_iter, 'acc')[0][1]
    train_score = svrg_model.score(train_iter, 'acc')[0][1]
    print(val_score, train_score)
    lr_params_svrg.update({lr: [val_score, train_score]})

lr_params

mlp_model.score(train_iter, 'acc')[0][1]

lr_params = {}
#Create a SVRGModule
for lr_ in list(range(100, 10, -20)):
    batch_size = 100
    train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
    lr = lr_/100
    print(lr)
    import logging
    logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
    # create a trainable module on compute context
    mlp_model = mx.mod.Module(symbol=mlp, context=ctx)
    mlp_model.fit(train_iter,  # train data
                  eval_data=val_iter,  # validation data
                  optimizer='sgd',  # use SGD to train
                  optimizer_params={'learning_rate': lr},  # use fixed learning rate
                  eval_metric='acc',  # report accuracy during training
                  batch_end_callback = mx.callback.Speedometer(batch_size, 1000), # output progress for each 100 data batches
                  num_epoch=5)  # train for at most 10 dataset passes
    val_score = mlp_model.score(val_iter, 'acc')[0][1]
    train_score = mlp_model.score(train_iter, 'acc')[0][1]
    print(val_score, train_score)
    lr_params.update({lr: [val_score, train_score]})

