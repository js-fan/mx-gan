import mxnet as mx
import numpy as np
import os


class Cifar10(mx.io.DataIter):
    def __init__(self, root, batch_size, train=True, shuffle=True):
        if train:
            files = [os.path.join(root, 'cifar-10-batches-py', 'data_batch_%d' %  i)\
                     for i in range(1, 6)]
            raw_data = [np.load(x) for x in files]
        else:
            raw_data = [np.load(os.path.join(root, 'cifar-10-batches-py', 'test_batch'))]
        self.data = np.concatenate([x['data'].reshape(-1, 3, 32, 32)\
                                    for x in raw_data], axis=0)
        self.label = np.array(sum([x['labels'] for x in raw_data], []))

        self.order = np.arange(self.data.shape[0])
        self.batch_size = batch_size
        self.num_batch = len(self.order) // self.batch_size
        self.shuffle = shuffle
        self.reset()

    def reset(self):
        self.current = 0
        if self.shuffle:
            np.random.permutation(self.order)

    def next(self):
        if self.current >= self.num_batch:
            raise StopIteration
        index = self.order[self.current * self.batch_size : (self.current+1) * self.batch_size]
        self.current += 1

        data = self.data[index].astype(np.float32) / 127.5 - 1.
        data = mx.nd.array(data)
        label = mx.nd.array(self.label[index])
        batch = mx.io.DataBatch(data=[data], label=[label])
        return batch
