import jax
import time
import jax.numpy as np
from HRR.with_flax import normal
from HRR.with_flax import Projection, Binding, Unbinding, CosineSimilarity

np.set_printoptions(precision=4, suppress=True)

batch = 32
features = 256

x = normal(shape=(batch, features), seed=0)
y = normal(shape=(batch, features), seed=1)

projection = Projection()
binding = Binding()
unbinding = Unbinding()
similarity = CosineSimilarity()

# create empty frozen dict as parameter less variable
var = projection.init(jax.random.PRNGKey(0), np.ones((batch, features)))

tic = time.time()
x = projection.apply(var, x)
y = projection.apply(var, y)

b = binding.apply(var, x, y)
y_ = unbinding.apply(var, b, x)

score = similarity.apply(var, y, y_)
toc = time.time()

print('y:', y[0])
print('y_prime:', y_[0])
print(f'score: {score[0]:.2f}')
print(f'Total time: {toc - tic:.4f}s')
