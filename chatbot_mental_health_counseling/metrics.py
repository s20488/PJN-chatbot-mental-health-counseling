import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import spacy
import torch


# nltk.download('vader_lexicon')
# nlp = spacy.load("en_core_web_sm")


# BLEU Metric
def calculate_bleu(references, candidates):
    reference_tokens = [[ref.split()] for ref in references]
    candidate_tokens = [cand.split() for cand in candidates]
    smoothing_function = SmoothingFunction().method1
    return corpus_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing_function)


# Perplexity Metric
def calculate_perplexity(text, tokenizer, model, device_map):
    encodings = tokenizer(text, return_tensors='pt')
    max_length = model.config.max_position_embeddings
    stride = 512

    lls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = i + stride
        trg_len = end_loc - i
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device_map[""])
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs.loss * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    return ppl.item()


# VADER Empathy Score
sia = SentimentIntensityAnalyzer()


def vader_empathy_score(text):
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']


# Dialog Quality Metrics
def dialog_quality_metrics(text):
    return {
        "length": len(text.split()),
        "unique_words": len(set(text.split()))
    }


# Relevance Score
def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.text for token in doc if token.pos_ == "NOUN"]
    return keywords


def relevance_score(text, prompt):
    prompt_keywords = extract_keywords(prompt)
    text_keywords = extract_keywords(text)
    score = sum(1 for word in prompt_keywords if word in text_keywords) / len(prompt_keywords)
    return score
