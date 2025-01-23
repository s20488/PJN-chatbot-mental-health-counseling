import json

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    logging,
)
from peft import PeftModel
import evaluate
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.bleu_score import corpus_bleu

# Загружаем необходимые ресурсы для VADER
nltk.download('vader_lexicon')

# Инициализируем анализатор VADER
sia = SentimentIntensityAnalyzer()

# Загружаем spaCy модель
nlp = spacy.load("en_core_web_sm")

model_name = "NousResearch/Llama-2-7b-chat-hf"
new_model = "Llama-2-7b-chat-finetune-qlora"

# model_name = "unsloth/Llama-3.2-1B-Instruct"
# new_model = "Llama-3.2-1B-Instruct-finetune-qlora"
#
# model_name = "JackFram/llama-68m"
# new_model = "llama-68m-finetune-qlora"

device_map = {"": 0}

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Run text generation pipeline with our next model
prompt = "How can I get to a place where I can be content from day to day?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200, temperature=0.6, top_p=0.85,
                no_repeat_ngram_size=2, eos_token_id=tokenizer.eos_token_id, do_sample=True)
result = pipe(f"<s>[INST] {prompt} [/INST]")

generated_text = result[0]["generated_text"]
print(f"Generated Text: {generated_text}")

# Метрика BLEU
def calculate_bleu(references, candidates):
    reference_tokens = [[ref.split()] for ref in references]
    candidate_tokens = [cand.split() for cand in candidates]
    smoothing_function = SmoothingFunction().method1
    return corpus_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing_function)

file_path = 'combined_dataset.json'
with open(file_path, 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]

references = [item['Response'] for item in data]
candidates = [generated_text for _ in data]

bleu_score = calculate_bleu(references, candidates)
print(f"BLEU score: {bleu_score}")

# Метрика Perplexity
def calculate_perplexity(text):
    encodings = tokenizer(text, return_tensors='pt')
    max_length = model.config.max_position_embeddings
    stride = 512

    lls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = i + stride
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device_map[""])
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs.loss * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    return ppl.item()

perplexity_score = calculate_perplexity(generated_text)
print(f"Perplexity: {perplexity_score}")

# Используем VADER для оценки эмпатии
def vader_empathy_score(text):
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']  # Эмпатия будет оценена как общий композитный балл

empathy_score = vader_empathy_score(generated_text)
print(f"Empathy Score (VADER): {empathy_score}")

# Метрики качества диалога (например, длина ответа и уникальность слов)
dialog_quality_metrics = {
    "length": len(generated_text.split()),
    "unique_words": len(set(generated_text.split()))
}
print(f"Dialog Quality Metrics: {dialog_quality_metrics}")

# Обновленная метрика релевантности с учетом извлеченных ключевых слов
def extract_keywords(text):
    doc = nlp(text)
    # Извлекаем существительные как ключевые слова
    keywords = [token.text for token in doc if token.pos_ == "NOUN"]
    return keywords

def relevance_score(text, prompt):
    # Извлекаем ключевые слова из текста и запроса
    prompt_keywords = extract_keywords(prompt)
    text_keywords = extract_keywords(text)

    # Считаем совпадения
    score = sum(1 for word in prompt_keywords if word in text_keywords) / len(prompt_keywords)
    return score

relevance = relevance_score(generated_text, prompt)
print(f"Relevance Score: {relevance}")
