from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Загрузка базовой модели
base_model = "JackFram/llama-160m"
model = AutoModelForCausalLM.from_pretrained(base_model)

# Загрузка адаптера
adapter_path = "/llama_mental_health_adapter"  # Путь к директории с `adapter_model.safetensors`
model = PeftModel.from_pretrained(model, adapter_path)

# Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Использование модели
input_text = "I have been dealing with depression and anxiety for a number of years. I have been on medication, but lately my depression has felt worse. Can counseling help?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
output = model.generate(input_ids, max_length=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
