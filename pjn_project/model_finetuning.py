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

# Logowanie
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# model_name = "NousResearch/Llama-2-7b-chat-hf"
# new_model = "Llama-2-7b-chat-finetune-qlora"

# model_name = "unsloth/Llama-3.2-1B-Instruct"
# new_model = "Llama-3.2-1B-Instruct-finetune-qlora"

model_name = "JackFram/llama-68m"
new_model = "llama-68m-finetune-qlora"

output_dir = f"./results-{new_model}"

# ----------------------------------------------------------------------------
# Zbiór danych:
# - Liczba tekstów - 3,512
# - Format - JSON
# - Anotacje - Każda próbka zawiera dwie kolumny tekstowe: "Context" i "Response"
#
# Podział na zbiór train, val i test:
# - train_dataset – 80% danych.
# - val_dataset – 10% danych.
# - test_dataset – 10% danych.
#
# Przetwarzanie wstępne:
# Funkcja preprocess_function łączy dwa elementy - Context i Response -
# w jeden ciąg tekstowy, dodając odpowiednie metki, takie jak "<s>[INST]...[/INST] </s>"
#
# Augmentacja danych:
# - Brak
# ----------------------------------------------------------------------------

# Załaduj zbiór danych z pliku combined_dataset.json
dataset = load_dataset("json", data_files="combined_dataset.json", split="train")

# Podział na zbiór treningowy, walidacyjny i testowy
train_test_split = dataset.train_test_split(test_size=0.1)
train_val_split = train_test_split['train'].train_test_split(test_size=0.1)

train_dataset = train_val_split['train']
val_dataset = train_val_split['test']
test_dataset = train_test_split['test']

# Przetwarzanie zbioru danych, łączenie "Context" i "Response"
def preprocess_function(examples):
    return {
        "text": f"<s>[INST] {examples['Context']} [/INST] {examples['Response']} </s>"
    }

train_dataset = train_dataset.map(preprocess_function)
val_dataset = val_dataset.map(preprocess_function)
test_dataset = test_dataset.map(preprocess_function)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=getattr(torch, "float16"),  # Załaduj tokenizer i model z konfiguracją QLoRA
    bnb_4bit_use_double_quant=False,
)

# ----------------------------------------------------------------------------
# Model i architektura:
# Architektura - Transformer
# Użyty model - Llama-2-7b-chat-hf/Llama-3.2-1B-Instruct/llama-68m
#
# Modyfikacje i dostrojenie modelu:
# 1. LoRA - technika adaptacji dużych modeli do specyficznych zadań
#    przy minimalnym koszcie obliczeniowym
# 2. Kwantowanie 4-bitowe - zmniejszenie rozmiaru modelu, oszczędność pamięci
#    i przyspieszenie obliczeń
# 4. Użycie SFTTrainer - fine-tuning modelu w zadaniach generowania tekstu
# ----------------------------------------------------------------------------

# Załaduj bazowy model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

# Załaduj tokenizer LLama
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Załaduj konfigurację QLoRA
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# Ustawienia treningu
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,
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
)

# SFT Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    args=training_arguments,
    processing_class=tokenizer,
)

# Rozpocznij trening
trainer.train()

# Zapisz wytrenowany model
trainer.model.save_pretrained(new_model)
