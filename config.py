config = {
    "use_big_model": False,
    "model_7b" : "LeoLM/leo-hessianai-7b-chat",
    "model_13b": "LeoLM/leo-hessianai-13b-chat",
    "tokenizer": "LeoLM/leo-hessianai-7b-chat",
    "num_steps": 0, # Anzahl der Samples im Trainingsdatensatz
    "batch_size": 8,
    "epochs": 2,
    "learning_rate": 3e-5, #(1e-5)
    "warmup_steps": 100,
    "n_freeze": 0,
    "grad_acc_steps": 1,
    "grad_checkpointing": False,
    "logging_steps": 10,
    "eval_examples": 2,
    "adam_betas": (0.9, 0.95), #(0.9, 0.99)
    "adam_eps": 1e-7,
    "adam_weight_decay": 0.01,
    "scheduler_eta_min": 1e-7,
    "use_lora": False,
    "lora_r": 64, # LoRA Parameter aus Empfehlungen aus Paper
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "task_type": "CAUSAL_LM",
    "target_modules" : [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
}