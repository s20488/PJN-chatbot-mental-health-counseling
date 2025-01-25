import json

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    logging,
)
from peft import PeftModel
from chatbot_mental_health_counseling.metrics import calculate_bleu, vader_empathy_score, dialog_quality_metrics, relevance_score, \
    calculate_perplexity

# model_name = "NousResearch/Llama-2-7b-chat-hf"
# new_model = "Llama-2-7b-chat-finetune-qlora"
#
model_name = "unsloth/Llama-3.2-1B-Instruct"
new_model = "Llama-3.2-1B-Instruct-finetune-qlora"
#
# model_name = "JackFram/llama-68m"
# new_model = "llama-68m-finetune-qlora"

# Suppress warnings
logging.set_verbosity(logging.CRITICAL)

# Reload the model in FP16 and merge with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

# Reload the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Run the text generation pipeline with the model
prompt = "Every winter I find myself getting sad because of the weather. How can I fight this?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200, temperature=0.6, top_p=0.85,
                no_repeat_ngram_size=2, eos_token_id=tokenizer.eos_token_id, do_sample=True)
result = pipe(f"<s>[INST] {prompt} [/INST]")

generated_text = result[0]["generated_text"]
print(f"Generated Text: {generated_text}")

# BLEU
file_path = 'combined_dataset.json'
with open(file_path, 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]

references = [item['Response'] for item in data]
candidates = [generated_text for _ in data]

bleu_score = calculate_bleu(references, candidates)
print(f"BLEU score: {bleu_score}")

# Perplexity Metric
perplexity_score = calculate_perplexity(generated_text, tokenizer, model, {"": 0})
print(f"Perplexity: {perplexity_score}")

# Empathy Score
empathy_score = vader_empathy_score(generated_text)
print(f"Empathy Score (VADER): {empathy_score}")

# Dialog Quality Metrics
dialog_metrics = dialog_quality_metrics(generated_text)
print(f"Dialog Quality Metrics: {dialog_metrics}")

# Relevance Score
relevance = relevance_score(generated_text, prompt)
print(f"Relevance Score: {relevance}")

# Save metrics results to a file
results_file_path = f'metrics-results-{new_model}.json'
metrics_results = {
    "bleu_score": bleu_score,
    "perplexity_score": perplexity_score,
    "empathy_score": empathy_score,
    "dialog_quality_metrics": dialog_metrics,
    "relevance_score": relevance
}

with open(results_file_path, 'w', encoding='utf-8') as results_file:
    json.dump(metrics_results, results_file, ensure_ascii=False, indent=4)
