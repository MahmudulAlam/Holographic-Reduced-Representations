import torch
import matplotlib.pyplot as plt
from HRR.with_pytorch import normal, projection_2d, binding_2d, unbinding_2d_approx

x = plt.imread('figs/image.jpg')
x = torch.tensor(x) / 255.
y = normal(x.shape, seed=0)
y = projection_2d(y, dim=(0, 1))


def normalize(x):
    """ normalization for visualization purposes only """
    min_ = torch.min(x)
    max_ = torch.max(x)
    return (x - min_) / (max_ - min_)


plt.figure(figsize=[12, 4])
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = 'lato'
plt.rcParams['font.weight'] = 'light'
plt.rcParams['axes.linewidth'] = 2.0

plt.subplot(1, 3, 1)
plt.imshow(normalize(x))
plt.title('Original Image')
plt.axis('off')

bind = binding_2d(x, y, dim=(0, 1))
plt.subplot(1, 3, 2)
plt.imshow(normalize(bind))
plt.title('Bound Image')
plt.axis('off')

x_prime = unbinding_2d_approx(bind, y, dim=(0, 1))
plt.subplot(1, 3, 3)
plt.imshow(normalize(x_prime))
plt.title('Retrieved Image')
plt.axis('off')

plt.savefig('figs/viz.jpg', bbox_inches='tight')
plt.show()
