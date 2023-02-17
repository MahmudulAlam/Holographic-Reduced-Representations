__version__ = '1.2.0'

try:
    from HRR.with_pytorch import *
    from HRR.real.with_pytorch import *
except ImportError:
    pass

try:
    from HRR.with_jax import *
    from HRR.with_flax import *
    from HRR.real.with_jax import *
    from HRR.real.with_flax import *
except ImportError:
    pass

try:
    from HRR.with_tensorflow import *
    from HRR.real.with_tensorflow import *
except ImportError:
    pass
