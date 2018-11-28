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

# LOGISTIC REGRESSION SGD

# define dataset and dataloader
mnist = mx.test_utils.get_mnist()

train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True,
                               data_name='data', label_name='softmax_label')
test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size * 2, shuffle=False,
                              data_name='data', label_name='softmax_label')

def sgd_logistic(train_iter, test_iter, hyperparams=None):
    # TODO: Define hyperparams API
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
    # TODO: Define returns