import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import random
import numpy as np
from dotenv import load_dotenv

# Отключаем предупреждения о параллелизме токенизатора
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Загрузка токена из .env файла
load_dotenv()
token = os.getenv("HUGGINGFACE_TOKEN")

if not token:
    raise ValueError("Hugging Face токен не найден. Проверьте .env файл!")

# Установка seed для воспроизводимости
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Загрузка данных
dataset = load_dataset("json", data_files="combined_dataset.json")

train_test_split = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_validation_split = train_test_split["train"].train_test_split(test_size=0.1, seed=42)

train_dataset = train_validation_split["train"]
validation_dataset = train_validation_split["test"]

# Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    token=token
)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

# Предобработка данных
def preprocess_data(example):
    return {
        "input_ids": tokenizer(
            example["Context"],
            truncation=True,
            padding="max_length",
            max_length=1024
        )["input_ids"],
        "labels": tokenizer(
            example["Response"],
            truncation=True,
            padding="max_length",
            max_length=1024
        )["input_ids"],
    }

train_dataset = train_dataset.map(preprocess_data, batched=True)
validation_dataset = validation_dataset.map(preprocess_data, batched=True)

train_dataset = train_dataset.remove_columns(["Context", "Response"])
validation_dataset = validation_dataset.remove_columns(["Context", "Response"])

# Загрузка модели
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    token=token,
    device_map="auto",
    torch_dtype=torch.float16
)

model.resize_token_embeddings(len(tokenizer))

# Настройка PEFT (LoRA)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

# Настройка гиперпараметров обучения
training_args = TrainingArguments(
    output_dir="./llama_results",
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=0.0002,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="steps",
    save_steps=500,
    eval_steps=500,
    save_total_limit=3,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    fp16=True,
    seed=42,
    dataloader_num_workers=4,
    report_to=[],
    optim="adamw_torch",
    load_best_model_at_end=True  # Включено
)

# Создание Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    processing_class=tokenizer,  # Замена устаревшего параметра tokenizer
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

# Обучение модели
trainer.train()

# Сохранение дообученного адаптера
adapter_save_path = "./llama_mental_health_adapter"
os.makedirs(adapter_save_path, exist_ok=True)
model.save_pretrained(adapter_save_path)
tokenizer.save_pretrained(adapter_save_path)

print(f"A model adapter was saved at: {adapter_save_path}")
