from HRR.with_pytorch import normal, projection, binding, unbinding, cosine_similarity
from HRR.with_pytorch import projection_2d, binding_2d, unbinding_2d

batch = 32
features = 256

print('** test: 01 single term **')
x = projection(normal(shape=(batch, features), seed=0))
y = projection(normal(shape=(batch, features), seed=1))

b = binding(x, y)
y_ = unbinding(b, x)

score = cosine_similarity(y, y_, dim=-1, keepdim=False)
print('y:', y[0])
print('y_prime:', y_[0])
print('score:', score[0])

print('** test: 02 multiple terms **')
x = projection(normal(shape=(batch, features), seed=0))
y = projection(normal(shape=(batch, features), seed=1))
w = projection(normal(shape=(batch, features), seed=2))
z = projection(normal(shape=(batch, features), seed=3))

b = binding(x, y) + binding(w, z)
y_ = unbinding(b, x)

score = cosine_similarity(y, y_, dim=-1, keepdim=False)
print('y:', y[0])
print('y_prime:', y_[0])
print('score:', score[0])

print('** test: 03 hierarchical **')
x = projection(normal(shape=(batch, features), seed=0))
y = projection(normal(shape=(batch, features), seed=1))
z = projection(normal(shape=(batch, features), seed=2))

b = binding(x, y)
c = binding(b, z)

b_ = unbinding(c, z)
y_ = unbinding(b_, x)

score = cosine_similarity(y, y_, dim=-1)
print('y:', y[0])
print('y_prime:', y_[0])
print('score:', score[0])

print('** test: 04 2D single term **')
x = projection_2d(normal(shape=(batch, 3, features, features), seed=0))
y = projection_2d(normal(shape=(batch, 3, features, features), seed=1))

b = binding_2d(x, y)
y_ = unbinding_2d(b, x)

score = cosine_similarity(y, y_, dim=[2, 3])
print('y:', y[0][0][0])
print('y_prime:', y_[0][0][0])
print('score:', score[0])

print('** test: 05 2D multiple term **')
x = projection_2d(normal(shape=(batch, 3, features, features), seed=0))
y = projection_2d(normal(shape=(batch, 3, features, features), seed=1))
w = projection_2d(normal(shape=(batch, 3, features, features), seed=2))
z = projection_2d(normal(shape=(batch, 3, features, features), seed=3))

b = binding_2d(x, y) + binding_2d(w, z)
y_ = unbinding_2d(b, x)

score = cosine_similarity(y, y_, dim=[2, 3])
print('y:', y[0][0][0])
print('y_prime:', y_[0][0][0])
print('score:', score[0])
