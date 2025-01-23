import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
import logging

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# model_name = "NousResearch/Llama-2-7b-chat-hf"
# new_model = "Llama-2-7b-chat-finetune-qlora"

# model_name = "unsloth/Llama-3.2-1B-Instruct"
# new_model = "Llama-3.2-1B-Instruct-finetune-qlora"

model_name = "JackFram/llama-68m"
new_model = "llama-68m-finetune-qlora"

lora_r = 64  # lora attention dimension/ rank
lora_alpha = 16  # lora scaling parameter
lora_dropout = 0.1  # lora dropout probability

use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

# output directory where the model predictions and checkpoints will be stored
output_dir = f"./results-{new_model}"

# number of training epochs
num_train_epochs = 5

# enable fp16/bf16 training (set bf16 to True when using A100 GPU in google colab)
fp16 = False
bf16 = True

# batch size per GPU for training
per_device_train_batch_size = 4

# batch size per GPU for evaluation
per_device_eval_batch_size = 4

# gradient accumulation steps - No of update steps
gradient_accumulation_steps = 1

# learning rate
learning_rate = 2e-4

# weight decay
weight_decay = 0.001

# Gradient clipping(max gradient Normal)
max_grad_norm = 0.3

# optimizer to use
optim = "paged_adamw_32bit"

# learning rate scheduler
lr_scheduler_type = "cosine"

# seed for reproducibility
seed = 1

# Number of training steps
max_steps = -1

# Ratio of steps for linear warmup
warmup_ratio = 0.03

# group sequnces into batches with same length
group_by_length = True

# save checkpoint every X updates steps
save_steps = 0

# Log at every X updates steps
logging_steps = 50

packing = False

# load the entire model on the GPU
device_map = {"": 0}

# Load dataset from combined_dataset.json
dataset = load_dataset("json", data_files="combined_dataset.json", split="train")

# Разделение на тренировочную, валидационную и тестовую выборки
train_test_split = dataset.train_test_split(test_size=0.1)
train_val_split = train_test_split['train'].train_test_split(test_size=0.1)

train_dataset = train_val_split['train']
val_dataset = train_val_split['test']
test_dataset = train_test_split['test']


# Preprocess dataset to combine `Context` and `Response`
def preprocess_function(examples):
    return {
        "text": f"<s>[INST] {examples['Context']} [/INST] {examples['Response']} </s>"
    }


train_dataset = train_dataset.map(preprocess_function)
val_dataset = val_dataset.map(preprocess_function)
test_dataset = test_dataset.map(preprocess_function)

# Load tokenizer and model with QLoRA config
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Checking GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        logger.info("=" * 80)
        logger.info("Your GPU supports bfloat16, you are getting accelerate training with bf16= True")
        logger.info("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLama tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load QLoRA config
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set Training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    weight_decay=weight_decay,
    lr_scheduler_type=lr_scheduler_type,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    max_steps=max_steps,
    logging_dir='./logs',
)

# SFT Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    args=training_arguments,
    processing_class=tokenizer,
)

# Start training
trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model)
