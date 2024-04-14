import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.debug.profiler as xp
import torch_xla.test.test_utils as test_utils
import torch_xla.distributed.spmd.xla_sharding as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd.xla_sharded_tensor import XLAShardedTensor
from torch_xla.distributed.spmd.xla_sharding import Mesh

xr.use_spmd()
assert xr.is_spmd() == True

device = xm.xla_device()
xm.master_print(xr.global_runtime_device_count())
xm.master_print(device)
