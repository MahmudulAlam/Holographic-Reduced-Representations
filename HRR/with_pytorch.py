import torch


def fft(x, dim):
    return torch.fft.fft(x, dim=dim)


def ifft(x, dim):
    return torch.fft.ifft(x, dim=dim)


def fft_2d(x, dim):
    return torch.fft.fft2(x, dim=dim)


def ifft_2d(x, dim):
    return torch.fft.ifft2(x, dim=dim)


def approx_inverse(x, dim):
    x = torch.flip(x, dims=[dim])
    return torch.roll(x, 1, dims=dim)


def approx_inverse_2d(x, dim):
    x = torch.flip(x, dims=dim)
    return torch.roll(x, (1, 1), dims=dim)


def inverse_2d(x, dim):
    x = ifft_2d(1. / fft_2d(x, dim), dim).real
    return torch.nan_to_num(x)


def projection(x, dim):
    f = torch.abs(fft(x, dim))
    p = ifft(fft(x, dim) / f, dim).real
    return torch.nan_to_num(p)


def projection_2d(x, dim):
    f = torch.abs(fft_2d(x, dim))
    p = ifft_2d(fft_2d(x, dim) / f, dim).real
    return torch.nan_to_num(p)


def binding(x, y, dim):
    s = ifft(torch.multiply(fft(x, dim), fft(y, dim)), dim)
    return s.real


def binding_2d(x, y, dim):
    b = ifft_2d(torch.multiply(fft_2d(x, dim), fft_2d(y, dim)), dim)
    return b.real


def unbinding(b, y, dim):
    yt = approx_inverse(y, dim)
    return binding(b, yt, dim)


def unbinding_2d(b, y, dim):
    yt = inverse_2d(y, dim)
    return binding_2d(b, yt, dim)


def unbinding_2d_approx(b, y, dim):
    yt = approx_inverse_2d(y, dim)
    return binding_2d(b, yt, dim)


def normal(shape, seed):
    torch.manual_seed(seed)
    d = torch.prod(torch.tensor(shape[1:]))
    return torch.normal(mean=0, std=1. / torch.sqrt(d), size=shape)


def inner_product(x, y, dim, keepdim=False):
    return torch.sum(x * y, dim=dim, keepdim=keepdim)


def cosine_similarity(x, y, dim=None, keepdim=False):
    if not dim:
        dim = list(range(-len(x.size()) // 2, 0))
    norm_x = torch.norm(x, dim=dim, keepdim=keepdim)
    norm_y = torch.norm(y, dim=dim, keepdim=keepdim)
    return torch.sum(x * y, dim=dim, keepdim=keepdim) / (norm_x * norm_y)


if __name__ == '__main__':
    x_ = normal(shape=(2, 3, 8, 8), seed=0)
    y_ = normal(shape=(2, 3, 8, 8), seed=1)

    x_ = projection(x_, dim=1)
    y_ = projection(y_, dim=1)

    bind = binding(x_, y_, dim=1)
    yp = unbinding(bind, x_, dim=1)

    score = cosine_similarity(y_, yp, dim=1)

    print(score)
