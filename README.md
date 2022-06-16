## Holographic Reduced Representations 🔥

<p align="center"><em>Supports</em></p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/169624977-b64f749d-01cf-4300-8e6f-9674bb1c56fc.png" height="60">
&nbsp;
  <img src="https://user-images.githubusercontent.com/37298971/169624973-a0d7e833-50ec-4e93-9a16-7701e975fe6e.png" height="60">
&nbsp;
  <img src="https://user-images.githubusercontent.com/37298971/169624976-ebf54b45-989f-4b70-af27-c75aee5060b5.png" height="60">
&nbsp;
  <img src="https://user-images.githubusercontent.com/37298971/169624975-d711dcc8-e590-491b-a3a5-055837487cf8.png" height="60">
</p>

<p align="center">
<a href="https://www.tensorflow.org">TensorFlow</a>&nbsp; &nbsp; &nbsp;<a href="https://pytorch.org">PyTorch</a>&nbsp; &nbsp; &nbsp;<a href="https://github.com/google/jax">JAX</a>&nbsp; &nbsp; &nbsp;<a href="https://github.com/google/flax">Flax</a>
</p>

<p align="justify">
Holographic Reduced Representations (HRR) is a method of representing compositional structures using circular convolution in distributed representations. The HRR operations <em>binding</em> and <em>unbinding</em> allow assigning abstract concepts to arbitrary numerical vectors. Given vectors x and y in a d-dimensional space, both can be combined using binding operation. Likewise, one of the vectors can be retrieved knowing one of the two vectors using unbinding operation.
</p>

To import the HRR library with the TensorFlow backend use ```HRR.with_tensorflow```, to import with the JAX backend use ```HRR.with_jax```, and so on. Vectors are sampled from a normal distribution with zero mean and the variance of the inverse of the dimension using ```normal``` function, with ```projection``` onto the ball of complex unit magnitude, to enforce that the inverse will be numerically stable during unbinding, proposed in [Learning with Holographic Reduced Representations](https://arxiv.org/pdf/2109.02157.pdf).

```python 
from HRR.with_pytorch import normal, projection, binding, unbinding, cosine_similarity


batch = 32
features = 256

x = projection(normal(shape=(batch, features), seed=0))
y = projection(normal(shape=(batch, features), seed=1))

b = binding(x, y)
y_prime = unbinding(b, x)

score = cosine_similarity(y, y_prime, dim=-1, keepdim=False)
print('score:', score[0])
# prints score: tensor(1.0000)
```

What makes HRR more interesting is that multiple vectors can be combined by element-wise addition of the vectors,
however, retrieval accuracy will decrease.

```python
x = projection(normal(shape=(batch, features), seed=0))
y = projection(normal(shape=(batch, features), seed=1))
w = projection(normal(shape=(batch, features), seed=2))
z = projection(normal(shape=(batch, features), seed=3))

b = binding(x, y) + binding(w, z)
y_prime = unbinding(b, x)

score = cosine_similarity(y, y_prime, dim=-1, keepdim=False)
print('score:', score[0])
# prints score: tensor(0.7483)
```

More interestingly, vectors can be combined and retrieved in hierarchical order. 🌳 

```
x    y
 \  /
  \/
b=x#y  z 
   \  /
    \/
 c=(x#y)#z
```

```python 
x = projection(normal(shape=(batch, features), seed=0))
y = projection(normal(shape=(batch, features), seed=1))
z = projection(normal(shape=(batch, features), seed=2))

b = binding(x, y)
c = binding(b, z)

b_ = unbinding(c, z)
y_ = unbinding(b_, x)

score = cosine_similarity(y, y_, dim=-1)
print('score:', score[0])
# prints score: tensor(1.0000)
```

### Flax Module (Fastest) ⚡ 
HRR library supports vector binding/unbinding as a Flax module. Like any other Flax module, this needs to be initialized first and then execute using the apply method.

```python
from HRR.with_flax import normal, Projection, Binding, Unbinding, CosineSimilarity


x = normal(shape=(batch, features), seed=0)
y = normal(shape=(batch, features), seed=1)

projection = Projection()
binding = Binding()
unbinding = Unbinding()
similarity = CosineSimilarity()

# create empty frozen dict as parameter less variable
var = projection.init(jax.random.PRNGKey(0), np.ones((batch, features)))

x = projection.apply(var, x)
y = projection.apply(var, y)

b = binding.apply(var, x, y)
y_ = unbinding.apply(var, b, x)

score = similarity.apply(var, y, y_)
print(f'score: {score[0]:.2f}')
# prints score: 1.00
```
