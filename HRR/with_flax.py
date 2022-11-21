import jax
import jax.numpy as np
import flax.linen as nn


class FFT(nn.Module):
    @nn.compact
    def __call__(self, x, axis):
        return np.fft.fft(x, axis=axis)


class IFFT(nn.Module):
    @nn.compact
    def __call__(self, x, axis):
        return np.fft.ifft(x, axis=axis)


class ApproxInverse(nn.Module):
    @nn.compact
    def __call__(self, x, axis):
        x = np.flip(x, axis=axis)
        return np.roll(x, 1, axis=axis)


class Projection(nn.Module):
    def setup(self):
        self.fft = FFT()
        self.ifft = IFFT()

    @nn.compact
    def __call__(self, x, axis):
        f = self.fft(x, axis=axis)
        p = np.real(self.ifft(f / np.abs(f), axis=axis))
        return np.nan_to_num(p)


class Binding(nn.Module):
    def setup(self):
        self.fft = FFT()
        self.ifft = IFFT()

    @nn.compact
    def __call__(self, x, y, axis):
        b = self.ifft(self.fft(x, axis=axis) * self.fft(y, axis=axis), axis=axis)
        return np.real(b)


class Unbinding(nn.Module):
    def setup(self):
        self.approx_inverse = ApproxInverse()
        self.binding = Binding()

    @nn.compact
    def __call__(self, b, y, axis):
        yt = self.approx_inverse(y, axis=axis)
        return self.binding(b, yt, axis=axis)


class CosineSimilarity(nn.Module):
    @nn.compact
    def __call__(self, x, y, axis=-1, keepdims=False):
        norm_x = np.linalg.norm(x, axis=axis, keepdims=keepdims)
        norm_y = np.linalg.norm(y, axis=axis, keepdims=keepdims)
        return np.sum(x * y, axis=axis, keepdims=keepdims) / (norm_x * norm_y)


def normal(shape, seed=0):
    d = np.prod(np.asarray(shape[1:]))
    std = 1. / np.sqrt(d)
    return std * jax.random.normal(jax.random.PRNGKey(seed), shape, dtype=np.float32)
