import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel

base_model = "mistralai/Mistral-7B-Instruct-v0.2"
adapter = "./llama_mental_health_adapter"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    add_bos_token=True,
    trust_remote_code=True,
    padding_side='left'
)

# Create peft model using base_model and finetuned adapter
config = PeftConfig.from_pretrained(adapter)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                             load_in_4bit=True,
                                             device_map='auto',
                                             torch_dtype='auto')
model = PeftModel.from_pretrained(model, adapter)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Prompt content:
messages = [
    {"role": "user", "content": "Hey Connor! I have been feeling a bit down lately.I could really use some advice on how to feel better?"}
]

input_ids = tokenizer.apply_chat_template(conversation=messages,
                                          tokenize=True,
                                          add_generation_prompt=True,
                                          return_tensors='pt').to(device)
output_ids = model.generate(input_ids=input_ids, max_new_tokens=512, do_sample=True, pad_token_id=2)
response = tokenizer.batch_decode(output_ids.detach().cpu().numpy(), skip_special_tokens = True)

# Model response:
print(response[0])
