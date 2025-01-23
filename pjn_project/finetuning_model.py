import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, EarlyStoppingCallback, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import random
import numpy as np
from trl import SFTTrainer, SFTConfig

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
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False, legacy=False)

# Установим pad_token и padding_side, чтобы избежать ошибок
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Шаг 3: Форматирование данных с токенами <|im_start|> и <|im_end|>
def format_with_tokens(context, response):
    """
    Форматирует данные в виде подсказок с токенами <|im_start|> и <|im_end|>.
    """
    return (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{context}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )

def preprocess_data_with_tokens(example):
    """
    Преобразует каждый пример данных в формат подсказок с токенами.
    """
    formatted_prompt = format_with_tokens(example["Context"], example["Response"])
    tokenized = tokenizer(
        formatted_prompt, truncation=True, padding="max_length", max_length=512
    )
    return {
        "input_ids": tokenized["input_ids"],
        "labels": tokenized["input_ids"],  # Используется для обучения
    }

train_dataset = train_dataset.map(preprocess_data_with_tokens, batched=True)
validation_dataset = validation_dataset.map(preprocess_data_with_tokens, batched=True)

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

sft_training_args = SFTConfig(
    learning_rate=1e-4,  # Более высокая скорость обучения для больших батчей
    per_device_train_batch_size=128,  # Увеличенный размер батча
    per_device_eval_batch_size=128,
    gradient_accumulation_steps=1,  # Нет необходимости в аккумулировании
    lr_scheduler_type="cosine",
    num_train_epochs=5,  # Уменьшено для ускорения обучения
    logging_strategy="steps",
    save_strategy="steps",
    eval_strategy="steps",
    logging_steps=50,  # Логгировать реже для уменьшения IO-загрузки
    eval_steps=50,
    save_steps=50,
    warmup_steps=500,  # Длинный разогрев
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    weight_decay=0.05,  # Чуть ниже, чтобы не переобучиться
    save_total_limit=5,  # Сохранить больше моделей
    output_dir="./llama_results_optimized",
    overwrite_output_dir=True,
    logging_dir="./logs",
    seed=42,
    dataloader_num_workers=8,  # Использование нескольких потоков
    report_to=[],
    dataloader_pin_memory=True,
    fp16=True  # Смешанная точность
)

# Шаг 7: Создание DataCollator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Шаг 8: Создание SFTTrainer
sft_trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    args=sft_training_args,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

# Шаг 9: Обучение модели с помощью SFTTrainer
sft_trainer.train()

# Шаг 10: Сохранение дообученного адаптера
adapter_save_path = "./llama_mental_health_adapter_test"
model.save_pretrained(adapter_save_path)
tokenizer.save_pretrained(adapter_save_path)

print(f"A model adapter was saved at: {adapter_save_path}")
