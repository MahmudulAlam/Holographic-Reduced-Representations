import tensorflow as tf
from warnings import warn

warn('For real-valued FFT, the dimension needs to be even, and the odd dimension will be reduced to even.')

"""
Note: In TensorFlow, all fft operations will be applied to last dimension. 
"""


def fft(x):
    return tf.signal.rfft(x)


def ifft(x):
    return tf.signal.irfft(x)


def fft_2d(x):
    return tf.signal.rfft2d(x)


def ifft_2d(x):
    return tf.signal.irfft2d(x)


def approx_inverse(x):
    x = tf.reverse(x, axis=[-1])
    return tf.roll(x, 1, axis=-1)


def inverse_2d(x):
    x = ifft_2d(1. / fft_2d(x))
    return tf.where(tf.math.is_nan(x), 0., x)


def projection(x):
    fx = fft(x)
    p = ifft(fx / tf.cast(tf.abs(fx), dtype=tf.complex64))
    return tf.where(tf.math.is_nan(p), 0., p)


def projection_2d(x):
    fx = fft_2d(x)
    p = ifft_2d(fx / tf.cast(tf.abs(fx), dtype=tf.complex64))
    return tf.where(tf.math.is_nan(p), 0., p)


def binding(x, y):
    return ifft(fft(x) * fft(y))


def binding_2d(x, y):
    return ifft_2d(tf.multiply(fft_2d(x), fft_2d(y)))


def unbinding(s, y):
    yt = approx_inverse(y)
    return binding(s, yt)


def unbinding_2d(b, y):
    yt = inverse_2d(y)
    return binding_2d(b, yt)


def normal(shape, seed):
    tf.random.set_seed(seed)
    d = tf.reduce_prod(tf.convert_to_tensor(shape[1:], dtype=tf.float32))
    return tf.random.normal(shape=shape, mean=0., stddev=1. / tf.sqrt(d))


def inner_product(x, y, axis=-1, keepdims=False):
    return tf.reduce_sum(x * y, axis=axis, keepdims=keepdims)


def cosine_similarity(x, y, axis=-1, keepdims=None):
    norm_x = tf.norm(x, axis=axis, keepdims=keepdims)
    norm_y = tf.norm(y, axis=axis, keepdims=keepdims)
    return tf.reduce_sum(x * y, axis=axis, keepdims=keepdims) / (norm_x * norm_y)


""" aliases """
convolve1d = binding
convolve2d = binding_2d

if __name__ == '__main__':
    x_ = normal(shape=(4, 8), seed=0)
    y_ = normal(shape=(4, 8), seed=1)

    x_ = projection(x_)
    y_ = projection(y_)

    bind = binding(x_, y_)
    yp = unbinding(bind, x_)
    print(y_.shape)
    print(yp.shape)
    score = cosine_similarity(y_, yp, axis=-1)

    print(score)
