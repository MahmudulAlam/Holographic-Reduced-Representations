__version__ = '1.0.1'

try:
    from HRR.with_pytorch import *
except ImportError:
    pass

try:
    from HRR.with_jax import *
    from HRR.with_flax import *
except ImportError:
    pass

try:
    from HRR.with_tensorflow import *
except ImportError:
    pass
