import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Загрузка базовой модели
base_model = "JackFram/llama-68m"
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)

# Загрузка адаптера (укажи правильный путь к адаптеру)
adapter_path = "./llama_mental_health_adapter_test"  # Обнови путь, если адаптер сохранён в другом месте
model = PeftModel.from_pretrained(model, adapter_path)

# Перевод модели в режим оценки
model.eval()

# Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Установим pad_token, чтобы избежать ошибок при генерации
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Перемещение модели на устройство (GPU или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Текст для генерации ответа
user_input = "I have been dealing with depression and anxiety for a number of years. I have been on medication, but lately my depression has felt worse. Can counseling help?"

# Форматирование текста с использованием токенов <|im_start|> и <|im_end|>
input_text = (
    f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    f"<|im_start|>user\n{user_input}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)

# Токенизация входного текста
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

# Автоматический расчет max_length
# Длина входного текста + запас на ответ
input_length = input_ids.shape[1]
response_length_buffer = 50  # Запас для ответа
max_length = input_length + response_length_buffer

print(f"Входная длина: {input_length}, max_length: {max_length}")

# Генерация текста с заданными параметрами
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=0.7,  # Управляет разнообразием ответов
        top_k=4,  # Использует только 4 вероятных токена
        repetition_penalty=1.2,  # Штраф за повторение
        penalty_alpha=0.5,  # Альфа штраф
        do_sample=True  # Включает сэмплирование
    )

# Расшифровка и вывод результата
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Извлечение ответа ассистента (после <|im_start|>assistant)
response_start = generated_text.find("<|im_start|>assistant") + len("<|im_start|>assistant\n")
response_end = generated_text.find("<|im_end|>", response_start)
final_response = generated_text[response_start:response_end].strip()

print("Сгенерированный ответ:")
print(final_response)
