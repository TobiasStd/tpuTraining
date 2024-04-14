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

# nur als weitere Metrik, nicht zu stark drauf achten
class Accuracy:
    def __init__(self):
        self.count = 0
        self.tp = 0.
    def update(self, logits, labels):
        logits = logits.argmax(dim=-1).view(-1).cpu()
        labels = labels.view(-1).cpu()
        tp = (logits == labels).sum()
        self.count += len(logits)
        self.tp += tp
        return tp / len(logits)
    def compute(self):
        return self.tp / self.count

# Da Fehler beim Importieren, einfach Code von HuggingFace kopiert
InputDataClass = NewType("InputDataClass", Any)

def torch_default_data_collator(features: List[InputDataClass]) -> Dict[str, Any]:

    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch

def default_data_collator(features: List[InputDataClass], return_tensors="pt") -> Dict[str, Any]:
    return torch_default_data_collator(features)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    xm.master_print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
