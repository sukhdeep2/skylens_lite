from skylens_args import *
from skylens import *

from skylens.parse_input import *

from jax.config import config
config.update('jax_disable_jit', True)

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('jax').setLevel(logging.WARNING)


Skylens_kwargs=parse_python(file_name='./tests/skylens_args.py')

Skylens_kwargs['do_xi']=False
Skylens_kwargs['do_pseudo_cl']=True

kappa0=Skylens(**Skylens_kwargs)

G=kappa0.cl_tomo()
client=client_get(Skylens_kwargs['scheduler_info'])
cc=client.compute(G['stack']).result()
