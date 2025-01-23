import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    logging,
)
from peft import PeftModel
import evaluate

model_name = "NousResearch/Llama-2-7b-chat-hf"
new_model = "Llama-2-7b-chat-finetune-qlora"

# model_name = "unsloth/Llama-3.2-1B-Instruct"
# new_model = "Llama-3.2-1B-Instruct-finetune-qlora"

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
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=150, temperature=0.7, top_p=0.9, no_repeat_ngram_size=2, eos_token_id=tokenizer.eos_token_id)
result = pipe(f"<s>[INST] {prompt} [/INST]")
generated_text = result[0]["generated_text"]
print(f"Generated Text: {generated_text}")

# Метрика BLEU
bleu_metric = evaluate.load("bleu")
references = [
    ["Your question is a fascinating one! As humans we have the ability to reflect on situations in our lives. Even if nothing currently goes on in a particular moment, it’s possible you’re reflecting on a serious or upsetting matter. And, our emotions linger within us. Just because a particular moment feels calm, inside your feelings may be the sense of a strong unsettled emotion from the recent past. Good for you to be aware of your own sensitivity to living with awareness of your moods and thoughts."],
    ["One thing that comes to mind is making a list of some things that happen each day. It could be that there are things that are affecting how upset you are, but because so many other things are going on, you may not notice. Another idea to try is to keep a list for a month of one good thing that happened each day. This way, when you're having a rough day, you have a list to think of and take a look at. Are you eating and sleeping in ways that are typical for you (typically at least two meals per day and roughly 8 hours of sleep that night (may be different depending on your age)? These two ideas are closely related to changes in your mood. From where do you have support? Friends or family? Can you take 5 or 10 minutes per day to do something that you enjoy? If you think back to the last time that you felt \"content,\" what was contributing to that? Another possibility is to try to be mindful of things that you do every day. For example, rather than eating a turkey sandwich as fast as possible on your lunch break, consider actually tasting it and enjoying it. Also consider giving yourself praise for doing something well. For example, when you finish your paperwork, take a moment to notice that and maybe reward yourself by checking your e-mail, reading five pages of a book, or something else that can be done quickly before you get back to your next task."],
    ["It's important to take a look inside and see what's going on with you to cause you to have these feelings. Please contact us in whatever way is most comfortable for you and we can get you set up with someone who will help you figure out this space in your life."]
]
predictions = [generated_text]
bleu_score = bleu_metric.compute(predictions=predictions, references=[references])
print(f"BLEU score: {bleu_score['bleu']}")

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

# Метрики эмпатии (пример: использование эмпатичных слов)
empathy_words = [
    "understand", "feel", "sorry", "empathize", "compassion", "care",
    "concern", "sympathy", "empathy", "support", "listen", "acknowledge",
    "appreciate", "comfort", "help", "kindness", "patience", "respect",
    "sensitive", "thoughtful"
]
empathy_score = sum(word in generated_text.lower() for word in empathy_words) / len(empathy_words)
print(f"Empathy Score: {empathy_score}")

# Метрики качества диалога (пример: длина ответа и уникальность слов)
dialog_quality_metrics = {
    "length": len(generated_text.split()),
    "unique_words": len(set(generated_text.split()))
}
print(f"Dialog Quality Metrics: {dialog_quality_metrics}")


# A/B тестирование (пример: сравнение двух моделей)
def ab_test(model_a, model_b, prompt):
    pipe_a = pipeline(task="text-generation", model=model_a, tokenizer=tokenizer, max_length=200)
    pipe_b = pipeline(task="text-generation", model=model_b, tokenizer=tokenizer, max_length=200)

    result_a = pipe_a(f"<s>[INST] {prompt} [/INST]")
    result_b = pipe_b(f"<s>[INST] {prompt} [/INST]")

    return result_a[0]["generated_text"], result_b[0]["generated_text"]


def relevance_score(text, prompt):
    # Считаем релевантность текста по ключевым словам, которые должны быть в ответе
    keywords = ["content", "day", "place", "happiness", "feel", "calm"]
    score = sum(1 for word in keywords if word in text.lower()) / len(keywords)
    return score


relevance = relevance_score(generated_text, prompt)
print(f"Relevance Score: {relevance}")
