import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Загрузка базовой модели
base_model = "JackFram/llama-68m"
model = AutoModelForCausalLM.from_pretrained(base_model)

# Загрузка адаптера (укажи правильный путь к адаптеру)
adapter_path = "./llama_mental_health_adapter"  # Обнови путь, если адаптер сохранён в другом месте
model = PeftModel.from_pretrained(model, adapter_path)

# Перевод модели в режим оценки
model.eval()

# Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Перемещение модели на устройство (GPU или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Текст для генерации ответа
input_text = "I have been dealing with depression and anxiety for a number of years. I have been on medication, but lately my depression has felt worse. Can counseling help?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

# Генерация текста с penalty_alpha и top_k
output = model.generate(
    input_ids,
    max_length=50,
    temperature=0.7,        # Управляет разнообразием ответов
    top_k=4,                # Использует только 4 вероятных токена
    penalty_alpha=0.5,      # Штраф за повторение
    do_sample=True          # Включает сэмплирование
)

# Расшифровка и вывод результата
print("Сгенерированный ответ:")
print(tokenizer.decode(output[0], skip_special_tokens=True))
