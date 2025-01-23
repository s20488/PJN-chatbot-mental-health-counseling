import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, EarlyStoppingCallback, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import random
import numpy as np
from trl import SFTTrainer, SFTConfig  # Импортируем SFTConfig

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
            example["Context"], truncation=True, padding="max_length", max_length=512
        )["input_ids"],
        "labels": tokenizer(
            example["Response"], truncation=True, padding="max_length", max_length=512
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
training_args = SFTConfig(
    learning_rate=2e-6,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    logging_strategy="steps",
    save_strategy="steps",
    evaluation_strategy="steps",
    logging_steps=10,
    eval_steps=10,
    save_steps=10,
    warmup_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    weight_decay=0.01,
    save_total_limit=10,
    output_dir="./llama_results_test",
    overwrite_output_dir=True,
    logging_dir="./logs",
    seed=42,
    dataloader_num_workers=24,
    report_to=[],
    dataloader_pin_memory=True
)

# Шаг 7: Создание DataCollator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Шаг 8: Создание SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    max_seq_length=2048,
    packing=True,
    args=training_args,
    data_collator=data_collator,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.005
        ),
    ],
)

# Шаг 9: Обучение модели
trainer.train()

# Шаг 10: Сохранение дообученного адаптера
adapter_save_path = "./llama_mental_health_adapter_test"
model.save_pretrained(adapter_save_path)
tokenizer.save_pretrained(adapter_save_path)

print(f"A model adapter was saved at: {adapter_save_path}")
