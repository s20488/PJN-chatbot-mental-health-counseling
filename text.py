from transformers import AutoTokenizer

# Загрузка токенизатора
base_model = "JackFram/llama-68m"
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Текст, который нужно токенизировать
text = "I have been dealing with depression and anxiety for a number of years."

# Токенизация текста
tokens = tokenizer(text, return_tensors="pt", truncation=False)

# Количество токенов
num_tokens = tokens.input_ids.size(1)
print(f"Количество токенов в тексте: {num_tokens}")
