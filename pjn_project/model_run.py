import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel

# Параметры модели и адаптера
base_model = "mistralai/Mistral-7B-Instruct-v0.2"
adapter = "./llama_mental_health_adapter"

# Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    add_bos_token=True,
    trust_remote_code=True,
    padding_side='left'
)

# Установка pad_token, если не установлен
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

# Загрузка модели и адаптера
config = PeftConfig.from_pretrained(adapter)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    load_in_4bit=True,
    device_map='auto',
    torch_dtype='auto'
)
model = PeftModel.from_pretrained(model, adapter)

# Увеличение размера токенов для поддержки новых токенов
model.resize_token_embeddings(len(tokenizer))

# Устройство для вычислений
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Пример контента (prompt)
messages = [
    {"role": "user", "content": "Hey Connor! I have been feeling a bit down lately. I could really use some advice on how to feel better?"}
]

# Токенизация ввода с генерацией attention_mask
input_text = tokenizer.apply_chat_template(
    conversation=messages,
    tokenize=False
)
tokenized_input = tokenizer(
    input_text,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=1024  # Максимальная длина ввода
).to(device)

# Генерация ответа модели
output_ids = model.generate(
    input_ids=tokenized_input["input_ids"],
    attention_mask=tokenized_input["attention_mask"],  # Передача attention_mask
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,  # Контроль случайности генерации
    top_p=0.9,        # Контроль для nucleus sampling
    pad_token_id=tokenizer.pad_token_id
)

# Декодирование ответа
response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

# Вывод ответа модели
print(response[0])
