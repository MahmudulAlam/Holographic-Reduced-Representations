import time
from HRR.with_jax import normal, projection, binding, unbinding, cosine_similarity
from HRR.with_jax import projection_2d, binding_2d, unbinding_2d

tic = time.time()
batch = 4
features = 16
axis = -1

print('** test: 01 single term **')
x = projection(normal(shape=(batch, features), seed=0), axis=axis)
y = projection(normal(shape=(batch, features), seed=1), axis=axis)

b = binding(x, y, axis=axis)
y_ = unbinding(b, x, axis=axis)

score = cosine_similarity(y, y_, axis=-1, keepdims=False)
print('y:', y[0])
print('y_prime:', y_[0])
print('score:', score[0])
toc = time.time()
print(f'Total time: {toc - tic:.4f}s')

print('** test: 02 multiple terms **')
x = projection(normal(shape=(batch, features), seed=0), axis)
y = projection(normal(shape=(batch, features), seed=1), axis)
w = projection(normal(shape=(batch, features), seed=4), axis)
z = projection(normal(shape=(batch, features), seed=5), axis)

b = binding(x, y, axis) + binding(w, z, axis)
y_ = unbinding(b, x, axis)

score = cosine_similarity(y, y_, axis=-1, keepdims=False)
print('y:', y[0])
print('y_prime:', y_[0])
print('score:', score[0])

print('** test: 03 hierarchical **')
x = projection(normal(shape=(batch, features), seed=0), axis)
y = projection(normal(shape=(batch, features), seed=1), axis)
z = projection(normal(shape=(batch, features), seed=2), axis)

b = binding(x, y, axis)
c = binding(b, z, axis)

b_ = unbinding(c, z, axis)
y_ = unbinding(b_, x, axis)

score = cosine_similarity(y, y_, axis=axis)
print('y:', y[0])
print('y_prime:', y_[0])
print('score:', score[0])

print('** test: 04 2D single term **')
x = projection_2d(normal(shape=(batch, 3, features, features), seed=0), axis=(-2, -1))
y = projection_2d(normal(shape=(batch, 3, features, features), seed=1), axis=(-2, -1))

b = binding_2d(x, y, axis=(-2, -1))
y_ = unbinding_2d(b, x, axis=(-2, -1))

score = cosine_similarity(y, y_, axis=(-2, -1))
print('y:', y[0][0][0])
print('y_prime:', y_[0][0][0])
print('score:', score[0])

print('** test: 05 2D multiple term **')
x = projection_2d(normal(shape=(batch, 3, features, features), seed=0), axis=(-2, -1))
y = projection_2d(normal(shape=(batch, 3, features, features), seed=1), axis=(-2, -1))
w = projection_2d(normal(shape=(batch, 3, features, features), seed=2), axis=(-2, -1))
z = projection_2d(normal(shape=(batch, 3, features, features), seed=3), axis=(-2, -1))

b = binding_2d(x, y, axis=(-2, -1)) + binding_2d(w, z, axis=(-2, -1))
y_ = unbinding_2d(b, x, axis=(-2, -1))

score = cosine_similarity(y, y_, axis=(-2, -1))
print('y:', y[0][0][0])
print('y_prime:', y_[0][0][0])
print('score:', score[0])
