import json
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    logging,
)
from peft import PeftModel
from trl import SFTTrainer

import metrics

# model_name = "NousResearch/Llama-2-7b-chat-hf"
# new_model = "Llama-2-7b-chat-finetune-qlora"
#
# model_name = "unsloth/Llama-3.2-1B-Instruct"
# new_model = "Llama-3.2-1B-Instruct-finetune-qlora"
#
model_name = "JackFram/llama-68m"
new_model = "llama-68m-finetune-qlora"

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

# Create the text-generation pipeline
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=200,
    temperature=0.6,
    top_p=0.85,
    no_repeat_ngram_size=2,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=True
)

# Load references (for BLEU)
file_path = '../combined_dataset.json'
with open(file_path, 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]

references = [item['Response'] for item in data]

# Accumulators for metrics
acc_bleu = 0.0
acc_perplexity = 0.0
acc_empathy = 0.0
avg_dialog_quality = None
acc_relevance = 0.0
count_prompts = 0

# Prompt input loop
while True:
    prompt = input("\nEnter your prompt (or 'quit'/'exit' to finish): ").strip()
    if prompt.lower() in ["quit", "exit"]:
        break
    if not prompt:
        continue

    result = pipe(f"<s>[INST] {prompt} [/INST]")
    generated_text = result[0]["generated_text"]
    print(generated_text)

    candidates = [generated_text for _ in data]
    bleu_score = metrics.calculate_bleu(references, candidates)
    perplexity_score = metrics.calculate_perplexity(generated_text, tokenizer, model, {"": 0})
    empathy_score = metrics.vader_empathy_score(generated_text)
    dq_score = metrics.dialog_quality_metrics(generated_text)
    relevance = metrics.relevance_score(generated_text, prompt)

    acc_bleu += bleu_score
    acc_perplexity += perplexity_score
    acc_empathy += empathy_score
    avg_dialog_quality = dq_score
    acc_relevance += relevance
    count_prompts += 1

# Compute and save average metrics
if count_prompts > 0:
    avg_bleu = acc_bleu / count_prompts
    avg_perplexity = acc_perplexity / count_prompts
    avg_empathy = acc_empathy / count_prompts
    avg_relevance = acc_relevance / count_prompts
    average_metrics = {
        "bleu_score": avg_bleu,
        "perplexity_score": avg_perplexity,
        "empathy_score": avg_empathy,
        "dialog_quality_metrics": avg_dialog_quality,
        "relevance_score": avg_relevance
    }
else:
    average_metrics = {}

results_file_path = f"metrics-results-{new_model}.json"
with open(results_file_path, 'w', encoding='utf-8') as results_file:
    json.dump(average_metrics, results_file, ensure_ascii=False, indent=4)
