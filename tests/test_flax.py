import jax
import time
import jax.numpy as np
import flax.linen as nn
from HRR.with_flax import normal
from HRR.with_flax import Projection, Binding, Unbinding, CosineSimilarity

np.set_printoptions(precision=4, suppress=True)

batch = 32
features = 1024

x = normal(shape=(batch, features), seed=0)
y = normal(shape=(batch, features), seed=1)


class Model(nn.Module):
    def setup(self):
        self.binding = Binding()
        self.unbinding = Unbinding()
        self.projection = Projection()
        self.similarity = CosineSimilarity()

    @nn.compact
    def __call__(self, x, y, axis):
        x = self.projection(x, axis=axis)
        y = self.projection(y, axis=axis)

        b = self.binding(x, y, axis=axis)
        y_ = self.unbinding(b, x, axis=axis)

        return self.similarity(y, y_, axis=axis, keepdims=False)


model = Model()
init_value = {'x': np.ones_like(x), 'y': np.ones_like(y), 'axis': -1}
var = model.init(jax.random.PRNGKey(0), **init_value)

tic = time.time()
inputs = {'x': x, 'y': y, 'axis': -1}
score = model.apply(var, **inputs)
toc = time.time()

print(score)
print(f'score: {score[0]:.2f}')
print(f'Total time: {toc - tic:.4f}s')
