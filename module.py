import mxnet as mx

class GANModule(object):
    def __init__(self, generator, discriminator, code_shape, data_shape, batch_size,
                 context=None):
        self.sym_gen = generator
        self.sym_dis = discriminator
        self.code_shape = code_shape
        self.data_shape = data_shape
        self.batch_size = batch_size
        self.context = [mx.cpu()] if context is None else context

        self.init_models()
        self.buffer_grads = {}
        self.update = self.update_basic

    def init_models(self):
        self.mod_gen = mx.mod.Module(
                symbol = self.sym_gen,
                data_names = ('data',),
                label_names = None,
                context = self.context)
        self.mod_gen.bind(data_shapes=[('data', (self.batch_size,)+self.data_shape)])

        self.mod_dis = mx.mod.Module(
                symbol = self.sym_dis,
                data_names = ('data',),
                label_names = None,
                context = self.context)
        self.mod_dis.bind(
                data_shapes = [('data', (self.batch_size,)+self.code_shape)],
                inputs_need_grad = True)

    def init_parameters(self, *args, **kwargs):
        self.mod_gen.init_params(*args, **kwargs)
        self.mod_dis.init_params(*args, **kwargs)

    def init_optimizer(self, *args, **kwargs):
        self.mod_gen.init_optimizer(*args, **kwargs)
        self.mod_dis.init_optimizer(*args, **kwargs)

    def cache_grads(self, mod, method='w'):
        '''
            'w': write (override)
            'a': add
            'r': read  (override)
            'ra': read and add
        '''
        if mod not in self.buffer_grads:
            assert method in ['w', 'a'], method
            self.buffer_grads[mod] = [[g.copyto(g.context) for g in grad] \
                                        for grad in mod._exec_group.grad_arrays]
            return
        if method == 'w':
            for grad_src, grad_dst in zip(mod._exec_group.grad_arrays, self.buffer_grads[mod]):
                for g_src, g_dst in zip(grad_src, grad_dst):
                    g_src.copyto(g_dst)
        elif method == 'a':
            for grad_src, grad_dst in zip(mod._exec_group.grad_arrays, self.buffer_grads[mod]):
                for g_src, g_dst in zip(grad_src, grad_dst):
                    g_dst += g_src
        elif method == 'r':
            for grad_src, grad_dst in zip(mod._exec_group.grad_arrays, self.buffer_grads[mod]):
                for g_src, g_dst in zip(grad_src, grad_dst):
                    g_dst.copyto(g_src)
        elif method == 'ra':
            for grad_src, grad_dst in zip(mod._exec_group.grad_arrays, self.buffer_grads[mod]):
                for g_src, g_dst in zip(grad_src, grad_dst):
                    g_src += g_dst
        else:
            raise ValueError("Unknown method {}".format(method))

    def get_code_batch(self):
        batch = mx.io.DataBatch(data=[
            mx.nd.normal(0, 1, (self.batch_size,)+self.code_shape)])
        return batch

    def update_basic(self, batch):
        # generate images
        code_batch = self.get_code_batch()
        self.mod_gen.forward(code_batch)
        batch_gen = mx.io.DataBatch(data=self.mod_gen.get_outputs())

        # update discriminator
        self.mod_dis.forward(batch)
        pred = mx.nd.sigmoid(self.mod_dis.get_outputs()[0])
        self.mod_dis.backward([pred - 1])
        self.cache_grads(self.mod_dis, 'w')

        self.mod_dis.forward(batch_gen)
        pred = mx.nd.sigmoid(self.mod_dis.get_outputs()[0])
        self.mod_dis.backward([pred])
        self.cache_grads(self.mod_dis, 'ra')
        self.mod_dis.update()

        # update generator
        self.mod_dis.forward(batch_gen)
        pred = mx.nd.sigmoid(self.mod_dis.get_outputs()[0])
        self.mod_dis.backward([pred - 1])
        grad_dis2gen = self.mod_dis.get_input_grads()[0]
        self.mod_gen.backward([grad_dis2gen])
        self.mod_gen.update()

    def save_params(self, prefix, epoch):
        save_names = [x.format(prefix, epoch) for x in [
                        '{}_gen_{:04d}.params',
                        '{}_dis_{:04d}.params',
                        '{}_gen_{:04d}.states',
                        '{}_dis_{:04d}.states']]
        self.mod_gen.save_params(save_names[0])
        self.mod_dis.save_params(save_names[1])
        self.mod_gen.save_optimizer_states(save_names[2])
        self.mod_dis.save_optimizer_states(save_names[3])
        return save_names

    def generate_images(self, code_batch):
        self.mod_gen.forward(code_batch)
        gen_imgs = self.mod_gen.get_outputs()[0].copyto(mx.cpu())
        return gen_imgs

