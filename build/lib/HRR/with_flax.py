import jax
import jax.numpy as np
import flax.linen as nn


class FFT(nn.Module):
    @nn.compact
    def __call__(self, x):
        return np.fft.fft(x)


class IFFT(nn.Module):
    @nn.compact
    def __call__(self, x):
        return np.fft.ifft(x)


class ApproxInverse(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = np.flip(x, axis=-1)
        return np.roll(x, 1, axis=-1)


class Projection(nn.Module):
    def setup(self):
        self.fft = FFT()
        self.ifft = IFFT()

    @nn.compact
    def __call__(self, x):
        f = np.abs(self.fft(x))
        p = np.real(self.ifft(self.fft(x) / f))
        return np.nan_to_num(p)


class Binding(nn.Module):
    def setup(self):
        self.fft = FFT()
        self.ifft = IFFT()

    @nn.compact
    def __call__(self, x, y):
        b = self.ifft(self.fft(x) * self.fft(y))
        return np.real(b)


class Unbinding(nn.Module):
    def setup(self):
        self.approx_inverse = ApproxInverse()
        self.binding = Binding()

    @nn.compact
    def __call__(self, b, y):
        yt = self.approx_inverse(y)
        return self.binding(b, yt)


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
