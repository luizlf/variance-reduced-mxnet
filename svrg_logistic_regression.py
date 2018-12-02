from __future__ import print_function
import os
import logging
import mxnet as mx
import requests as rq


def download_svrg_module():

    try:
        os.makedirs('./svrg_optimization')
        init_svrg = rq.get(
            'https://raw.githubusercontent.com/apache/incubator-mxnet/master/python/mxnet/contrib/svrg_optimization/__init__.py')
        with open('./svrg_optimization/__init__.py', 'w') as file_init:
            file_init.write(init_svrg.text)

        modu_svrg = rq.get(
            'https://raw.githubusercontent.com/apache/incubator-mxnet/master/python/mxnet/contrib/svrg_optimization/svrg_module.py')
        with open('./svrg_optimization/svrg_module.py', 'w') as file_modu:
            file_modu.write(modu_svrg.text)

        opti_svrg = rq.get(
            'https://raw.githubusercontent.com/apache/incubator-mxnet/master/python/mxnet/contrib/svrg_optimization/svrg_optimizer.py')
        with open('./svrg_optimization/svrg_optimizer.py', 'w') as file_opti:
            file_opti.write(opti_svrg.text)

    except FileExistsError:
        pass


class LogisticRegression(object):

    def __init__(self, dataset, epochs, context, batch_size, is_svrg=True, optimizer='sgd', hyperparams=None, **kwargs):
        self.dataset = dataset.lower()
        self.epochs = epochs
        self.context = context
        self.hyperparams = hyperparams
        self.batch_size = batch_size
        self.is_svrg = is_svrg
        self.optimizer = optimizer
        self.train_iter = None
        self.test_iter = None

        self.define_dataset()

        # return super().__init__(dataset, epochs, context, hyperparams, **kwargs)

    def define_dataset(self):
        if self.dataset == 'mnist' or self.dataset is None:
            # define dataset and dataloader
            mnist = mx.test_utils.get_mnist()

            self.train_iter = mx.io.NDArrayIter(mnist['train_data'],
                                                mnist['train_label'],
                                                self.batch_size, shuffle=True, data_name='data',
                                                label_name='softmax_label')
            self.test_iter = mx.io.NDArrayIter(mnist['test_data'],
                                               mnist['test_label'],
                                               self.batch_size * 2, shuffle=False, data_name='data',
                                               label_name='softmax_label')

        elif self.dataset == 'cifar-10':
            mx.test_utils.get_cifar10()


    def run_logistic(self, update_freq=2):

        # data input and formatting
        data = mx.sym.var('data')  # (bs, 1, 28, 28) - MNIST
        label = mx.sym.var('softmax_label')
        data = mx.sym.Flatten(data)  # (bs, 28*28) - MNIST

        # logistic regression network
        fc = mx.sym.FullyConnected(data, num_hidden=10, name='fc')
        logist = mx.sym.SoftmaxOutput(fc, label=label, name='softmax')

        # metrics and eval
        metric_list = [mx.metric.Accuracy(output_names=['softmax_output'], label_names=['softmax_label']),
                       mx.metric.CrossEntropy(output_names=['softmax_output'], label_names=['softmax_label'])]
        eval_metrics = mx.metric.CompositeEvalMetric(metric_list)

        # create and 'compile' network
        if self.is_svrg:
            model = SVRGModule(symbol=logist,
                               data_names=['data'],
                               label_names=['softmax_label'],
                               context=self.context,
                               update_freq=update_freq)
        else:
            model = mx.mod.Module(logist,
                                  data_names=['data'],
                                  label_names=['softmax_label'],
                                  context=self.context)

        model.bind(data_shapes=self.train_iter.provide_data,
                   label_shapes=self.train_iter.provide_label)
        model.init_params()
        model.init_optimizer(kvstore='local',
                             optimizer=self.optimizer,
                             optimizer_params=self.hyperparams)

        logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

        model.fit(self.train_iter,  # train data
                  eval_data=self.test_iter,  # validation data
                  # optimizer='sgd',  # use SGD to train
                  # optimizer_params = (('learning_rate', lr),),  # use fixed learning rate
                  eval_metric=eval_metrics,  # report accuracy during training
                  # output progress for each 500 data batches
                  batch_end_callback=mx.callback.Speedometer(
                      batch_size, 500),
                  epoch_end_callback=mx.callback.do_checkpoint(
                      'logistic', 10),
                  num_epoch=epochs)  # train for at most <epochs> dataset passes
        val_score = model.score(self.train_iter, 'acc')[0][1]
        train_score = model.score(self.train_iter, 'acc')[0][1]
        print(val_score, train_score)
        exit()

        # TODO: Define function returns
        # IDEA: pandas df with all class params, timestamp, and scores. Save this to disk
        # lr_params_svrg.update({lr: [val_score, train_score]})

        return train_score


if __name__ == '__main__':
    try:
        from svrg_optimization.svrg_module import SVRGModule
    except ModuleNotFoundError:
        download_svrg_module()
        from svrg_optimization.svrg_module import SVRGModule

    run = True

    # define hyperparameters
    lr = 0.2
    batch_size = 32
    epochs = 100
    step = 300

    if run:
        ctx = mx.cpu()
        epochs = 10
        lr_svrg_sgd = LogisticRegression(
            dataset='MNIST', optimizer='sgd', epochs=epochs, context=ctx, batch_size=batch_size, is_svrg=True,
            hyperparams={'learning_rate': 0.2})
        lr_svrg_sgd.run_logistic(2)
