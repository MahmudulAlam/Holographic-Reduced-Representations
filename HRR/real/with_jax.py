import jax
import jax.numpy as np


def fft(x, axis):
    return np.fft.rfft(x, axis=axis)


def ifft(x, axis):
    return np.fft.irfft(x, axis=axis)


def fft_2d(x, axis):
    return np.fft.rfft2(x, axes=axis)


def ifft_2d(x, axis):
    return np.fft.irfft2(x, axes=axis)


def approx_inverse(x, axis):
    x = np.flip(x, axis=axis)
    return np.roll(x, 1, axis=axis)


def inverse_2d(x, axis):
    x = ifft_2d(1. / fft_2d(x, axis), axis)
    return np.nan_to_num(x)


def projection(x, axis):
    f = np.abs(fft(x, axis))
    p = ifft(fft(x, axis) / f, axis)
    return np.nan_to_num(p)


def projection_2d(x, axis):
    f = np.abs(fft_2d(x, axis))
    p = ifft_2d(fft_2d(x, axis) / f, axis)
    return np.nan_to_num(p)


def binding(x, y, axis):
    return ifft(fft(x, axis) * fft(y, axis), axis)


def binding_2d(x, y, axis):
    return ifft_2d(np.multiply(fft_2d(x, axis), fft_2d(y, axis)), axis)


def unbinding(s, y, axis):
    yt = approx_inverse(y, axis)
    return binding(s, yt, axis)


def unbinding_2d(b, y, axis):
    yt = inverse_2d(y, axis)
    return binding_2d(b, yt, axis)


def normal(shape, seed=0):
    d = np.prod(np.asarray(shape[1:]))
    std = 1. / np.sqrt(d)
    return std * jax.random.normal(jax.random.PRNGKey(seed), shape, dtype=np.float32)


def inner_product(x, y, axis, keepdims=False):
    return np.sum(x * y, axis=axis, keepdims=keepdims, )


def cosine_similarity(x, y, axis=None, keepdims=None):
    if not axis:
        axis = tuple(range(-len(x.size()) // 2, 0))
    norm_x = np.linalg.norm(x, axis=axis, keepdims=keepdims)
    norm_y = np.linalg.norm(y, axis=axis, keepdims=keepdims)
    return np.sum(x * y, axis=axis, keepdims=keepdims) / (norm_x * norm_y)


""" aliases """
convolve1d = binding
convolve2d = binding_2d

if __name__ == '__main__':
    x_ = normal(shape=(2, 4, 8, 8), seed=0)
    y_ = normal(shape=(2, 4, 8, 8), seed=1)

    x_ = projection(x_, axis=1)
    y_ = projection(y_, axis=1)

    bind = binding(x_, y_, axis=1)
    yp = unbinding(bind, x_, axis=1)

    score = cosine_similarity(y_, yp, axis=1)
    print(score[0])
