import mxnet as mx
import numpy as np
import cv2
from datetime import datetime
EPS = 1e-5

def get_time():
    return datetime.now().strftime('%m-%d %H:%M:%S')

def wrap_color(string, color):
    try:
        header = {
                'red':       '\033[91m',
                'green':     '\033[92m',
                'yellow':    '\033[93m',
                'blue':      '\033[94m',
                'purple':    '\033[95m',
                'cyan':      '\033[96m',
                'darkcyan':  '\033[36m',
                'bold':      '\033[1m',
                'underline': '\033[4m'}[color.lower()]
    except KeyError:
        raise ValueError("Unknown color: {}".format(color))
    return header + string + '\033[0m'



def info(logger, msg, color=None):
    msg = '[{}]'.format(get_time()) + msg
    logger.info(msg)

    if color is not None:
        msg = wrap_color(msg, color)
    print(msg)


def summary_args(logger, args, color=None):
    keys = args.keys()
    keys.sort()
    length = max([len(x) for x in keys])
    msg = [('{:<'+str(length)+'}: {}').format(k, args[k]) for k in keys]

    msg = '[{}]\n'.format(get_time()) + '\n'.join(msg)
    logger.info(msg)

    if color is not None:
        msg = wrap_color(msg, color)
    print(msg)


def save_image(src, x, norm=False, flip=False):
    # Input should be 4D-Tensor
    if isinstance(x, mx.nd.NDArray):
        x = x.asnumpy()

    h, w = x.shape[2:]
    num_img = x.shape[0]
    num_row = int(np.sqrt(num_img) + .5)
    num_col = (num_img + num_row - 1) // num_row
    shape = (num_row, num_col)
    space = (min(h, w) // 20,) * 2


    # Different input range:
    if norm:
        print("Normed Image Range from [{}, {}]".format(
              x.min(), x.max()))
        x -= x.min()
        x /= (x.max() if x.max() > 0 else 1)
        x *= 255
    else:
        if x.min() >= 0 and x.max() > 1:
            # [0, 255]
            print("Image Range set as [0, 255] for [{}, {}]".format(
                  x.min(), x.max()))
        elif x.min() >= 0 and x.max() <= 1:
            # [0, 1]
            print("Image Range set as [0, 1] for [{}, {}]".format(
                  x.min(), x.max()))
            x *= 255
        elif x.min() < 0 and x.max() > 0:
            # [-1, 1]
            print("Image Range set as [-1, 1] for [{}, {}]".format(
                  x.min(), x.max()))
            x = np.clip(x, a_min=-1, a_max=1)
            x = (x + 1) * 127.5
        else:
            raise ValueError("Unknown Image Range: [{}, {}]".format(
                x.min(), x.max()))

    x = x.astype(np.uint8)
    x = x.transpose(0, 2, 3, 1)
    out = np.ones((h * shape[0] + space[0] * (shape[0] + 1),
                   w * shape[1] + space[1] * (shape[1] + 1),
                   x.shape[3]), dtype=np.uint8) * 255

    for i, img in enumerate(x):
        row = i // shape[1]
        col = i % shape[1]
        if row >= shape[0]:
            warnings.warn("Given images are more that 'shape' specified: {} vs {}".format(x.shape[0], shape))
            break
        out[row*h+(row+1)*space[0] : (row+1)*(h+space[0]), col*w+(col+1)*space[1] : (col+1)*(w+space[1])] = img
    if flip:
        out = out[..., ::-1].astype(np.uint8)
    cv2.imwrite(src, out)
