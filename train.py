import mxnet as mx
import logging
import time
import os
from subprocess import call
from easydict import EasyDict
from symbols import generator32, discriminator32
from module import GANModule
from loader import Cifar10
from utils import summary_args, info, save_image


def config():
    model_name = 'cifar-10-gan'
    snapshot = 'snapshot/' + model_name

    root = 'data/cifar/' # the folder where 'cifar-10-batches-py' is
    data_shape = (3, 32, 32)
    code_shape = (64,)
    batch_size = 100
    num_epoch  = 100
    lr = 3e-4
    beta1 = 0.5

    frequence = 100
    max_save = 3
    gpsu = [0]
    return EasyDict(locals())
    
def run(args):
    # preparation
    if not os.path.exists(args.snapshot):
        os.makedirs(args.snapshot)
    logging.basicConfig(filename=os.path.join(args.snapshot, args.model_name+'.log'),
                        level=logging.INFO)
    logger = logging.getLogger()

    # data, symbol, model
    data_loader = Cifar10(args.root, args.batch_size, train=True, shuffle=True)
    sym_generator = generator32()
    sym_discriminator = discriminator32()

    mod = GANModule(sym_generator, sym_discriminator,
                    code_shape=tuple(args.code_shape),
                    data_shape=tuple(args.data_shape),
                    batch_size=args.batch_size,
                    context=[mx.gpu(x) for x in args.gpus])
    mod.init_params(mx.init.Normal(0.01))
    mod.init_optimizer(
            kvstore = 'device',
            optimizer = 'adam',
            optimizer_params = {
                'learning_rate': args.lr,
                'beta1': args.beta1})

    # summary before trainining
    summary_args(logger, args, color='green')
    const_code = mod.get_code_batch()
    const_batch = next(data_loader)
    save_images(os.path.join(args.snapshot, args.model_name+'_real.jpg'),
               const_batch.data[0], flip=True)
    checkpoints = []

    # train
    for n_epoch in range(args.num_epoch):
        data_loader.reset()
        tic = time.time()

        for n_batch, batch in enumerate(data_loader):
            mod.update(batch)

            if (n_batch + 1) % args.frequence == 0:
                msg = "Epoch={}, Batch={}, Speed={:.1f} b/s".format(
                        n_epoch, n_batch+1, args.frequence/(time.time()-tic))
                info(logger, msg)
                tic = time.time()

        # save checkpoint
        _checkpoints = mod.save_params(os.path.join(args.snapshot, args.model_name),
                                       n_epoch)
        info(logger, '\n'.join(["Saved Checkpoints:"]+["  "+x for x in _checkpoints]),
             'yellow')
        checkpoints += _checkpoints
        if len(checkpoints) > args.max_save * len(_checkpoints):
            call(['rm'] + checkpoints[:len(_checkpoints)])
            checkpoints = checkpoints[len(_checkpoints):]

        # generate some images
        gen_imgs = mod.generate_images(const_code)
        gen_imgs = mx.nd.clip(gen_imgs, a_min=-1, a_max=1)
        save_images(os.path.join(args.snapshot, args.model_name+'_{:04d}.jpg'.format(
                    n_epoch)), gen_imgs, flip=True)

if __name__ == '__main__':
    run(config())
