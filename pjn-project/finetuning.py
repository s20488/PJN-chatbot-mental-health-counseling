import json
import os
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

os.environ["TMPDIR"] = "/mnt/data/tmp"
os.environ["HF_DATASETS_CACHE"] = "/mnt/data/datasets_cache"
os.environ["HF_HOME"] = "/mnt/data/huggingface_cache"

# Select the model to fine-tune
# model_name = "NousResearch/Llama-2-7b-chat-hf"
# new_model = "Llama-2-7b-chat-finetune-qlora"
#
# model_name = "unsloth/Llama-3.2-1B-Instruct"
# new_model = "Llama-3.2-1B-Instruct-finetune-qlora"
#
model_name = "JackFram/llama-68m"
new_model = "llama-68m-finetune-qlora"

output_dir = f"./results-{new_model}"

# Load the dataset
dataset = load_dataset("json", data_files="combined_dataset.json", split="train")

# Split the dataset into train, validation, and test sets
dataset = dataset.train_test_split(test_size=0.2, seed=42)
test_valid = dataset['test'].train_test_split(test_size=0.5, seed=42)
dataset['validation'] = test_valid['train']
dataset['test'] = test_valid['test']


# Preprocess the data by combining "Context" and "Response" to create instructions
def preprocess_function(examples):
    return {
        "text": f"<s>[INST] {examples['Context']} [/INST] {examples['Response']} </s>"
    }


dataset = dataset.map(preprocess_function)

# 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=getattr(torch, "bfloat16"),
    bnb_4bit_use_double_quant=False,
)

# Load the base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, legacy=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# LoRA configuration for fine-tuning
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training arguments
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,
    remove_unused_columns=False,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=10,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    weight_decay=0.001,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    group_by_length=True,
    max_steps=-1,
    logging_dir='./logs',
    eval_strategy="steps",
    eval_steps=10,
)

# Initialize the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    peft_config=peft_config,
    args=training_arguments,
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Evaluate the model
trainer.evaluate()

# Save the fine-tuned model
trainer.model.save_pretrained(new_model)

# Test the model
test_results = trainer.predict(dataset['test'])

# Save the test results to a file
with open("test_results.json", "w") as f:
    json.dump(test_results, f)
