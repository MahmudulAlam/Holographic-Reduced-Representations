<h2 align="center"><a href="https://github.com/MahmudulAlam/Holographic-Reduced-Representations">Holographic Reduced Representations</a></h2>

[![GitHub issues](https://img.shields.io/github/issues/MahmudulAlam/Holographic-Reduced-Representations)](https://github.com/MahmudulAlam/Holographic-Reduced-Representations/issues)
[![GitHub forks](https://img.shields.io/github/forks/MahmudulAlam/Holographic-Reduced-Representations)](https://github.com/MahmudulAlam/Holographic-Reduced-Representations/network)
[![GitHub stars](https://img.shields.io/github/stars/MahmudulAlam/Holographic-Reduced-Representations)](https://github.com/MahmudulAlam/Holographic-Reduced-Representations/stargazers)
[![GitHub license](https://img.shields.io/github/license/MahmudulAlam/Holographic-Reduced-Representations)](https://github.com/MahmudulAlam/Holographic-Reduced-Representations/blob/main/LICENSE)
[![Version](https://img.shields.io/badge/version-1.1.0-f56207.svg?longCache=true&style=flat)]()
[![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2F)](https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2FMahmudulAlam%2FHolographic-Reduced-Representations%2F)

<p align="center"><b><em>Supports</em></b></p> 
<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/169624977-b64f749d-01cf-4300-8e6f-9674bb1c56fc.png" height="80">
&nbsp;
  <img src="https://user-images.githubusercontent.com/37298971/169624973-a0d7e833-50ec-4e93-9a16-7701e975fe6e.png" height="80">
&nbsp;
  <img src="https://user-images.githubusercontent.com/37298971/169624976-ebf54b45-989f-4b70-af27-c75aee5060b5.png" height="80">
&nbsp;
  <img src="https://user-images.githubusercontent.com/37298971/169624975-d711dcc8-e590-491b-a3a5-055837487cf8.png" height="80">
</p>

## Install 🎉

```
pip install hrr
```

<!-- <b>else</b>
``` 
pip install git+https://github.com/MahmudulAlam/Holographic-Reduced-Representations.git
``` -->

## Update :hammer_and_wrench:

- v1.2.3 - support for real-valued FFT, can be accessed by ```HRR.real```
- v1.1.0 - ```dim/axis``` support for PyTorch, JAX & Flax
- For TensorFlow binding/unbinding can only be applied to the last dimension

## Intro :studio_microphone:

<p align="justify">
Holographic Reduced Representations (HRR) is a method of representing compositional structures using circular convolution in distributed representations. The HRR operations <em>binding</em> and <em>unbinding</em> allow assigning abstract concepts to arbitrary numerical vectors. Given vectors x and y in a d-dimensional space, both can be combined using binding operation. Likewise, one of the vectors can be retrieved knowing one of the two vectors using unbinding operation.
</p>

## Docs :green_book:

HRR library supports <a href="https://www.tensorflow.org">TensorFlow</a>, <a href="https://pytorch.org">PyTorch</a>
, <a href="https://github.com/google/jax">JAX</a>, and <a href="https://github.com/google/flax">Flax</a>. To import the
HRR package with the TensorFlow backend use ```HRR.with_tensorflow```, to import with the JAX backend
use ```HRR.with_jax```, and so on. Vectors are sampled from a normal distribution with zero mean and the variance of the
inverse of the dimension using ```normal``` function, with ```projection``` onto the ball of complex unit magnitude, to
enforce that the inverse will be numerically stable during unbinding, proposed
in [Learning with Holographic Reduced Representations](https://arxiv.org/abs/2109.02157).

```python 
from HRR.with_pytorch import normal, projection, binding, unbinding, cosine_similarity


batch = 32
features = 256

x = projection(normal(shape=(batch, features), seed=0), dim=-1)
y = projection(normal(shape=(batch, features), seed=1), dim=-1)

b = binding(x, y, dim=-1)
y_prime = unbinding(b, x, dim=-1)

score = cosine_similarity(y, y_prime, dim=-1, keepdim=False)
print('score:', score[0])
# prints score: tensor(1.0000)
```

What makes HRR more interesting is that multiple vectors can be combined by element-wise addition of the vectors,
however, retrieval accuracy will decrease.

```python
x = projection(normal(shape=(batch, features), seed=0), dim=-1)
y = projection(normal(shape=(batch, features), seed=1), dim=-1)
w = projection(normal(shape=(batch, features), seed=2), dim=-1)
z = projection(normal(shape=(batch, features), seed=3), dim=-1)

b = binding(x, y, dim=-1) + binding(w, z, dim=-1)
y_prime = unbinding(b, x, dim=-1)

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
x = projection(normal(shape=(batch, features), seed=0), dim=-1)
y = projection(normal(shape=(batch, features), seed=1), dim=-1)
z = projection(normal(shape=(batch, features), seed=2), dim=-1)

b = binding(x, y, dim=-1)
c = binding(b, z, dim=-1)

b_ = unbinding(c, z, dim=-1)
y_ = unbinding(b_, x, dim=-1)

score = cosine_similarity(y, y_, dim=-1)
print('score:', score[0])
# prints score: tensor(1.0000)
```

## Flax Module (Fastest) ⚡

HRR package supports vector binding/unbinding as a Flax module. Like any other Flax module, this needs to be initialized
first and then execute using the apply method.

```python
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
# prints score: 1.00
# Total time: 0.0088s
```

## Processing :framed_picture:

```apply.py``` shows an example of how to apply binding/unbinding to an image. The bound image is the composite representation of the original image and another matrix sampled from a normal distribution performed by the ```binding``` operation. Using the ```unbinding``` operation, the original image can be retrieved without any loss. 

<p align="center">
  <img src="figs/viz.jpg" height="200">
</p>

## Papers :scroll:
[![Paper](https://img.shields.io/badge/CSPS-ICML%202022-098ae6.svg?longCache=true&style=flat)]()

🌱 [```Deploying Convolutional Networks on Untrusted Platforms Using 2D Holographic Reduced Representations```](http://arxiv.org/abs/2206.05893) [```GitHub```](https://github.com/NeuromorphicComputationResearchProgram/Connectionist-Symbolic-Pseudo-Secrets)

```bibtex 
@inproceedings{Alam2022,
  archivePrefix = {arXiv},
  arxivId = {2206.05893},
  author = {Alam, Mohammad Mahmudul and Raff, Edward and Oates, Tim and Holt, James},
  booktitle = {International Conference on Machine Learning},
  eprint = {2206.05893},
  title = {{Deploying Convolutional Networks on Untrusted Platforms Using 2D Holographic Reduced Representations}},
  url = {http://arxiv.org/abs/2206.05893},
  year = {2022}
}
```

[![Paper](https://img.shields.io/badge/Hrrformer-ICML%202023-098ae6.svg?longCache=true&style=flat)]()

🌱 [```Recasting Self-Attention with Holographic Reduced Representations```](https://arxiv.org/abs/2305.19534) [```GitHub```](https://github.com/NeuromorphicComputationResearchProgram/Hrrformer)

```bibtex
@inproceedings{alam2023recasting,
  title={Recasting self-attention with holographic reduced representations},
  author={Alam, Mohammad Mahmudul and Raff, Edward and Biderman, Stella and Oates, Tim and Holt, James},
  booktitle={International Conference on Machine Learning},
  pages={490--507},
  year={2023},
  organization={PMLR}
}
```

[![Paper](https://img.shields.io/badge/Subitizing-AAAI%20workshop%202024-098ae6.svg?longCache=true&style=flat)]()

🌱 [```Towards Generalization in Subitizing with Neuro-Symbolic Loss using Holographic Reduced Representations```](https://arxiv.org/abs/2312.15310) [```GitHub```](https://github.com/MahmudulAlam/Subitizing)

```bibtex
@article{alam2023towards,
  title={Towards generalization in subitizing with neuro-symbolic loss using holographic reduced representations},
  author={Alam, Mohammad Mahmudul and Raff, Edward and Oates, Tim},
  journal={arXiv preprint arXiv:2312.15310},
  year={2023}
}
```

[![Paper](https://img.shields.io/badge/HGConv-AISTATS%202024-098ae6.svg?longCache=true&style=flat)]()

🌱 [```Holographic Global Convolutional Networks for Long-Range Prediction Tasks in Malware Detection @ AISTATS 2024```](https://arxiv.org/abs/2403.17978) [```GitHub```](https://github.com/FutureComputing4AI/HGConv)

```bibtex
@inproceedings{alam2024holographic,
  title={Holographic Global Convolutional Networks for Long-Range Prediction Tasks in Malware Detection},
  author={Alam, Mohammad Mahmudul and Raff, Edward and Biderman, Stella R and Oates, Tim and Holt, James},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={4042--4050},
  year={2024},
  organization={PMLR}
}
```

[![Paper](https://img.shields.io/badge/HLB-NeurIPS%202024-098ae6.svg?longCache=true&style=flat)]()

🌱 [```A Walsh Hadamard Derived Linear Vector Symbolic Architecture @ NeurIPS 2024```](https://arxiv.org/abs/2410.22669) [```GitHub```](https://github.com/FutureComputing4AI/Hadamard-derived-Linear-Binding)

```bibtex
@article{alam2024walsh,
  title={A Walsh Hadamard Derived Linear Vector Symbolic Architecture},
  author={Alam, Mohammad Mahmudul and Oberle, Alexander and Raff, Edward and Biderman, Stella and Oates, Tim and Holt, James},
  journal={arXiv preprint arXiv:2410.22669},
  year={2024}
}
```

## Report 🐛🚧🚩

To report a bug or any other questions, please feel free to open an issue.


<h2 align="center">Thanks🌼</h2>
<p align="center">Thanks to <a href="https://twitter.com/EdwardRaffML" target="_blank"><img src="https://user-images.githubusercontent.com/37298971/206940177-a9661598-18a2-4534-ade9-2c517da7b8f7.png" height="14">@EdwardRaffML</a> and <a href="https://twitter.com/oatesbag" target="_blank"><img src="https://user-images.githubusercontent.com/37298971/206940177-a9661598-18a2-4534-ade9-2c517da7b8f7.png" height="14">@oatesbag</a> for their constant support to this research endeavor.</p>
