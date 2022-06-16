import torch


def fft(x):
    return torch.fft.fft(x)


def ifft(x):
    return torch.fft.ifft(x)


def fft_2d(x):
    return torch.fft.fft2(x)


def ifft_2d(x):
    return torch.fft.ifft2(x)


def approx_inverse(x):
    x = torch.flip(x, dims=[-1])
    return torch.roll(x, 1, dims=-1)


def inverse_2d(x):
    x = ifft_2d(1. / fft_2d(x)).real
    return torch.nan_to_num(x)


def projection(x):
    f = torch.abs(fft(x))
    p = ifft(fft(x) / f).real
    return torch.nan_to_num(p)


def projection_2d(x):
    f = torch.abs(fft_2d(x))
    p = ifft_2d(fft_2d(x) / f).real
    return torch.nan_to_num(p)


def binding(x, y):
    s = ifft(torch.multiply(fft(x), fft(y)))
    return s.real


def binding_2d(x, y):
    b = ifft_2d(torch.multiply(fft_2d(x), fft_2d(y)))
    return b.real


def unbinding(b, y):
    yt = approx_inverse(y)
    return binding(b, yt)


def unbinding_2d(b, y):
    yt = inverse_2d(y)
    return binding_2d(b, yt)


def normal(shape, seed):
    torch.manual_seed(seed)
    d = torch.prod(torch.tensor(shape[1:]))
    return torch.normal(mean=0, std=1. / torch.sqrt(d), size=shape)


def inner_product(x, y, dim=-1, keepdim=False):
    return torch.sum(x * y, dim=dim, keepdim=keepdim)


def cosine_similarity(x, y, dim=None, keepdim=False):
    if not dim:
        dim = list(range(-len(x.size()) // 2, 0))
    norm_x = torch.norm(x, dim=dim, keepdim=keepdim)
    norm_y = torch.norm(y, dim=dim, keepdim=keepdim)
    return torch.sum(x * y, dim=dim, keepdim=keepdim) / (norm_x * norm_y)
