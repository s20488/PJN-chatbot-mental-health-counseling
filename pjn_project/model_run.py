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

# Перемещение модели на устройство (GPU или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Текст для генерации ответа
input_text = "I have been dealing with depression and anxiety for a number of years. I have been on medication, but lately my depression has felt worse. Can counseling help?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

# Генерация текста с улучшенными параметрами
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=512,
        temperature=0.9,  # Управляет разнообразием ответов
        top_k=50,  # Использует 50 вероятных токенов
        top_p=0.9,  # Использует nucleus sampling
        repetition_penalty=1.0016,  # Штраф за повторение
        do_sample=True  # Включает сэмплирование
    )

# Расшифровка и вывод результата
print("Сгенерированный ответ:")
print(tokenizer.decode(output[0], skip_special_tokens=True))
