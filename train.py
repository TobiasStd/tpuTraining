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

from trainUtils import Accuracy
from trainUtils import default_data_collator
from trainUtils import print_trainable_parameters
from distribute import partition_module
from config import config

import log

xr.use_spmd()
assert xr.is_spmd() == True

logger = log.get_logger(__name__)

@torch.no_grad()
def eval(epoch, model, eval_dataloader):
    device = xm.xla_device()
    model.eval()
    eval_acc = Accuracy()
    loss = 0.
    steps = 0
    
    for _, batch in enumerate(pbar:=tqdm(eval_dataloader, leave=False)):
        pbar.set_description("evaluting epoch")
        gc.collect()
        
        # batch auf Device laden
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['labels'].to(device)
        
        xs.mark_sharding(input_ids, mesh, (0, 1))
        xs.mark_sharding(attention_mask, mesh, (0, 1))
        xs.mark_sharding(targets, mesh, (0, 1))
        
        # forward 
        predicted = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=targets
        )
        
        loss += predicted.loss
        eval_acc.update(predicted.logits, targets)
        steps += 1
    
    eval_loss = loss.item() / steps
    eval_acc = eval_acc.compute()
    logger.warning(f'evaluated: loss={eval_loss}, accuracy={eval_acc}\n')
    logger.warning(f'epoch {epoch+1} finished.')

    model.train()

def train(model, optimizer, scheduler, train_dataloader, eval_dataloader):
    device = xm.xla_device()
    model.train()
    steps = 0
    acc = Accuracy()
    
    for epoch in tqdm(range(config['epochs'])):
        
        xm.master_print(f'beginning with epoch {epoch+1}.')
        
        for step, batch in enumerate(train_dataloader):
            gc.collect()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device)

            xs.mark_sharding(input_ids, mesh, (0, 1))
            xs.mark_sharding(attention_mask, mesh, (0, 1))
            xs.mark_sharding(targets, mesh, (0, 1))
             
            predicted = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=targets
            )
            
            loss = predicted.loss / config['grad_acc_steps']
            
            loss.backward()
            
            if step % config['grad_acc_steps'] == 0:
                optimizer.step()
                xm.mark_step()
                scheduler.step()
                optimizer.zero_grad()

                if step % config['logging_steps'] == 0:
                    xm.master_print(f'Training: Epoche {epoch+1}, Step {steps+1}: loss={loss.item() * config["grad_acc_steps"]}, accuracy={acc.update(predicted.logits, targets)}')
                steps += 1
                
        eval(epoch, model, eval_dataloader)


if __name__ == "__main__":
    train_dataset = Dataset.load_from_disk("tpuTraining/train_dataset")
    eval_dataset = Dataset.load_from_disk("tpuTraining/eval_dataset")

    config["num_steps"] = len(train_dataset)
    xm.master_print(f"Anzahl der Trainingssamples: {config['num_steps']}")

    if config["use_big_model"]:
        model_id = config["model_13b"]
    else:
        model_id = config["model_7b"]
        
    tokenizer_id = config["tokenizer"]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    device = xm.xla_device()
    num_devices = xr.global_runtime_device_count()
    xm.master_print(device)

    logger.warning(num_devices)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    #tokenizer.pad_token = tokenizer.eos_token
    #tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 
    )

    logger.warning("Modell wurde geladen.")

    if config["use_lora"]:
        peft_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["target_modules"],
            task_type=config["task_type"],
            inference_mode=False,
        )
        model = get_peft_model(model, peft_config)
        xm.master_print("use lora fine tuning:")
        model.print_trainable_parameters()
    else:
        if config["use_big_model"]:
            cnt = 0
            for param in model.parameters():
                cnt += 1
                param.requires_grad = False
                if cnt > 270:
                    param.requires_grad = True
        else:
            xm.master_print(f"use full fine tuning, freeze {config['n_freeze']} layers:")
            for param in model.parameters(): param.requires_grad = False
            for param in model.lm_head.parameters(): param.requires_grad = True
            for param in model.model.layers[config["n_freeze"]:].parameters(): param.requires_grad = True
        print_trainable_parameters(model)

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices, 1)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ('dp', 'fsdp', 'mp'))
    partition_module(model, mesh)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        collate_fn=default_data_collator,
        shuffle=True
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config["batch_size"],
        collate_fn=default_data_collator,
        shuffle=False
    )

    num_iterations = int(config['num_steps'] / config['batch_size'] / num_devices)
    xm.master_print(f"Trainingsiterationen auf einem Device: {num_iterations}")

    optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config["learning_rate"], 
            betas=config["adam_betas"], 
            eps=config["adam_eps"], 
            weight_decay=config["adam_weight_decay"],
        )
        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_iterations*config["epochs"],
            eta_min=config["scheduler_eta_min"],
            last_epoch=-1,
            verbose=False
        )

    start_time = time.time()

    train(model, optimizer, scheduler, train_dataloader, eval_dataloader)

    xm.master_print(f'Vergangene Zeit: {time.time() - start_time}')

    xm.master_print("Modell jetzt auf HuggingFace hochladen...")

    model = model.cpu()

    model.push_to_hub(
        "Tobiiax/full-training-7b",
        tokenizer=tokenizer,
        commit_message="fully fine tuned LeoLM-Chat (7B)",
        private=False,
        token="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"
    )
