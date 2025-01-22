import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import random
import numpy as np

# Установка seed для воспроизводимости
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Шаг 1: Загрузка данных
dataset = load_dataset("json", data_files="combined_dataset.json")

# Разделение данных на тренировочную, валидационную и тестовую части
train_test_split = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_validation_split = train_test_split["train"].train_test_split(test_size=0.1, seed=42)

train_dataset = train_validation_split["train"]
validation_dataset = train_validation_split["test"]

print(f"Train size: {len(train_dataset)}, Validation size: {len(validation_dataset)}")

# Шаг 2: Загрузка токенизатора
base_model = "JackFram/llama-68m"
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)

# Установим pad_token, чтобы избежать ошибки
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Шаг 3: Предобработка данных
def preprocess_data(example):
    return {
        "input_ids": tokenizer(
            example["Context"], truncation=True, padding="max_length", max_length=256
        )["input_ids"],
        "labels": tokenizer(
            example["Response"], truncation=True, padding="max_length", max_length=256
        )["input_ids"],
    }

train_dataset = train_dataset.map(preprocess_data, batched=True)
validation_dataset = validation_dataset.map(preprocess_data, batched=True)

train_dataset = train_dataset.remove_columns(["Context", "Response"])
validation_dataset = validation_dataset.remove_columns(["Context", "Response"])

# Шаг 4: Загрузка модели
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.float16
)

# Шаг 5: Настройка PEFT (LoRA)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

# Шаг 6: Настройка гиперпараметров обучения
training_args = TrainingArguments(
    output_dir="./llama_results",
    overwrite_output_dir=True,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=5e-5,
    num_train_epochs=200,
    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="steps",
    save_steps=500,
    eval_steps=500,
    save_total_limit=3,
    warmup_ratio=0.2,
    lr_scheduler_type="cosine",
    fp16=True,
    seed=42,
    dataloader_num_workers=24,
    report_to=[],
    load_best_model_at_end=True
)

# Шаг 7: Создание Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    processing_class=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

# Шаг 8: Обучение модели
trainer.train()

# Шаг 9: Сохранение дообученного адаптера
adapter_save_path = "./llama_mental_health_adapter"
model.save_pretrained(adapter_save_path)
tokenizer.save_pretrained(adapter_save_path)

print(f"A model adapter was saved at: {adapter_save_path}")
