import jax
import jax.numpy as np


def fft(x):
    return np.fft.fft(x)


def ifft(x):
    return np.fft.ifft(x)


def fft_2d(x):
    return np.fft.fft2(x)


def ifft_2d(x):
    return np.fft.ifft2(x)


def approx_inverse(x):
    x = np.flip(x, axis=-1)
    return np.roll(x, 1, axis=-1)


def inverse_2d(x):
    x = ifft_2d(1. / fft_2d(x)).real
    return np.nan_to_num(x)


def projection(x):
    f = np.abs(fft(x))
    p = np.real(ifft(fft(x) / f))
    return np.nan_to_num(p)


def projection_2d(x):
    f = np.abs(fft_2d(x))
    p = ifft_2d(fft_2d(x) / f).real
    return np.nan_to_num(p)


def binding(x, y):
    b = ifft(fft(x) * fft(y))
    return np.real(b)


def binding_2d(x, y):
    b = ifft_2d(np.multiply(fft_2d(x), fft_2d(y)))
    return np.real(b)


def unbinding(s, y):
    yt = approx_inverse(y)
    return binding(s, yt)


def unbinding_2d(b, y):
    yt = inverse_2d(y)
    return binding_2d(b, yt)


def normal(shape, seed=0):
    d = np.prod(np.asarray(shape[1:]))
    std = 1. / np.sqrt(d)
    return std * jax.random.normal(jax.random.PRNGKey(seed), shape, dtype=np.float32)


def inner_product(x, y, axis=-1, keepdims=False):
    return np.sum(x * y, axis=axis, keepdims=keepdims, )


def cosine_similarity(x, y, axis=None, keepdims=None):
    if not axis:
        axis = tuple(range(-len(x.size()) // 2, 0))
    norm_x = np.linalg.norm(x, axis=axis, keepdims=keepdims)
    norm_y = np.linalg.norm(y, axis=axis, keepdims=keepdims)
    return np.sum(x * y, axis=axis, keepdims=keepdims) / (norm_x * norm_y)
