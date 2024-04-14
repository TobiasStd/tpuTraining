import torch
from datasets import Dataset
from torch.utils.data import DataLoader
import transformers
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import List, NewType, Any, Dict
from collections.abc import Mapping
import numpy as np
import time
import gc
from typing import List, NewType, Any, Dict
from collections.abc import Mapping
import numpy as np
import re
from peft import LoraConfig, get_peft_model

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.debug.profiler as xp
import torch_xla.test.test_utils as test_utils
import torch_xla.distributed.spmd.xla_sharding as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd.xla_sharded_tensor import XLAShardedTensor
from torch_xla.distributed.spmd.xla_sharding import Mesh

partition_specs = (
        ("model\\.embed_tokens", ("mp", "fsdp")),
        ("self_attn\\.(q_proj|k_proj|v_proj)", ("fsdp", "mp")),
        ("self_attn\\.o_proj", ("mp", "fsdp")),
        ("mlp\\.gate_proj", ("fsdp", "mp")),
        ("mlp\\.down_proj", ("mp", "fsdp")),
        ("mlp\\.up_proj", ("fsdp", "mp")),
        ("lm_head", ("fsdp", "mp")),
    )
    
strkey2id = {
    "dp": 0,
    "fsdp": 1, 
    "mp": 2 
}

def partition_module(model, mesh, device='xla', verbose=True):
    model.to(device)

    for name, module in (tqdm(model.named_modules(), desc="partitioning model", disable=not verbose, position=0)):
        if not hasattr(module, "weight") or not isinstance(module.weight, torch.nn.Parameter):
            continue
        
        find = False
        
        for rule_pattern, spec in partition_specs:
            if re.findall(rule_pattern, name):
                if verbose:
                    print("match", rule_pattern, name, spec)
                
                xs.mark_sharding(module.weight, mesh, spec)
                find = True
                break
            
        if not find:
            if verbose:
                xm.master_print(f"no match {module}", name, module.weight.size(), module.weight.dim())
            xs.mark_sharding(module.weight, mesh, tuple([None] * module.weight.dim()))
