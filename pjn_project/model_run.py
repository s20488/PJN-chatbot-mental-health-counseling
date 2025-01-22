import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Загрузка базовой модели
base_model = "JackFram/llama-68m"
model = AutoModelForCausalLM.from_pretrained(base_model)

# Загрузка адаптера
adapter_path = "./llama_mental_health_adapter"
model = PeftModel.from_pretrained(model, adapter_path)

# Перевод модели в режим оценки
model.eval()

# Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained(base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Перемещение модели на устройство (GPU или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Текст для генерации ответа
input_text = (
    "I have been dealing with depression and anxiety for a number of years. "
    "I have been on medication, but lately my depression has felt worse. Can counseling help?"
)

input_ids = tokenizer(
    input_text, return_tensors="pt", truncation=True, max_length=512
).input_ids.to(device)

# Генерация текста с оптимизированными параметрами для длинного ответа
with torch.no_grad():  # Для экономии памяти
    output = model.generate(
        input_ids,
        max_length=512,            # Максимальная длина текста
        temperature=0.8,           # Баланс разнообразия
        top_k=50,                  # Расширение выбора
        top_p=0.9,                 # Нучное сэмплирование
        penalty_alpha=0.6,         # Штраф за повторение
        repetition_penalty=1.2,    # Дополнительный штраф за повторы
        no_repeat_ngram_size=3,    # Запрет на повторение фраз из 3 слов
        do_sample=True             # Сэмплирование
    )

# Расшифровка и вывод результата
print("Сгенерированный ответ:")
print(tokenizer.decode(output[0], skip_special_tokens=True))
