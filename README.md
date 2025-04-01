# Project: Chatbot for Mental Health Counseling

## Dataset
- **Number of texts**: 3,512
- **Format**: JSON
- **Annotations**: Each sample contains two text columns: **"Context"** and **"Response"**
- **Dataset link:**: [Mental Health Counseling Conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations)

### Train/Val/Test Split:
The dataset is loaded in its entirety and used for model training.

### Preprocessing:
The  `preprocess_function` combines  **Context** and **Response** into a single text string, adding special tokens such as: `<s>[INST] {context} [/INST] {response} </s>`

---
## Model and Architecture
- **Architecture**: Transformer
- **Models used**: 
  - **LLaMA-2-7b-chat-hf**
  - **LLaMA-3.2-1B-Instruct**
  - **LLaMA-68m**

### Modifications and Fine-Tuning:
1. **LoRA**: A technique for adapting large models to specific tasks with minimal computational overhead.
2. **QLoRA**: 4-bit quantization, which reduces model size, saves memory, and speeds up computations.

### Inspiration:
- [Fine-tuning LLaMA-2 using LoRA and QLoRA - A Comprehensive Guide](https://medium.com/@harsh.vardhan7695/fine-tuning-llama-2-using-lora-and-qlora-a-comprehensive-guide-fd2260f0aa5f)
---
## Methodology

### Preprocessing:
- **AutoTokenizer** splits text into tokens, converts them to numerical IDs, and formats prompts as:
  `"<s>[INST] {prompt} [/INST]"`

### Training:
- The **LLaMA** model is **fine-tuning** on the data using **LoRA** and **QLoRA** techniques.

### Evaluation:
- Text quality is assessed using the following metrics:
  - **BLEU**
  - **Perplexity**
  - **VADER Empathy Score**
  - **Relevance Score**

### Models and Prompting Approaches:
- **Models used**: LLaMA-2-7b-chat-hf, LLaMA-3.2-1B-Instruct, LLaMA-68m
- **Instruction-based prompting**: The model responds to textual instructions, generating answers based on provided prompts.
---
## Model Quality Metrics:
1. **BLEU**: Measures the quality of generated text against reference responses.
   - **Acceptance threshold**: BLEU > 0.3
2. **Perplexity**: Measures how well the model predicts the next word in a sequence.
   - **Acceptance threshold**: Perplexity < 0.5
3. **VADER Empathy Score**: Evaluates the emotional tone of generated text.
   - **Acceptance threshold**: Empathy score > 0.5
4. **Dialog Quality Metrics**: Assesses response length and unique word count to determine detail and diversity.
   - **Acceptance criteria**: 
   - **Length**: Appropriately long responses.
   - **Unique words**: Higher counts indicate greater response variety.
5. **Relevance Score**: Measures how strongly the generated text addresses the query.
   - **Acceptance threshold**: Relevance score > 0.7

### Results:
- **LLaMA-2-7b-chat-hf**
```json
{
    "bleu_score": 0.006609087401414352,
    "perplexity_score": 2.5055453777313232,
    "empathy_score": 0.7476,
    "dialog_quality_metrics": {
        "length": 123,
        "unique_words": 99
    },
    "relevance_score": 1.0
}
```
- **LLaMA-3.2-1B-Instruct**
```json
{
    "bleu_score": 0.012234656162563733,
    "perplexity_score": 3.347900629043579,
    "empathy_score": 0.9406,
    "dialog_quality_metrics": {
        "length": 175,
        "unique_words": 112
    },
    "relevance_score": 1.0
}
```
- **LLaMA-68m**
```json
{
    "bleu_score": 0.019426576877632835,
    "perplexity_score": 5.947700023651123,
    "empathy_score": 0.9846,
    "dialog_quality_metrics": {
        "length": 164,
        "unique_words": 103
    },
    "relevance_score": 1.0
}
```
---
## Additional Libraries:
- **Transformers**
- **Peft**
- **SpaCy**
- **NLTK**
- **Torch**

## Hardware and Software Environment:
- **GPU**: NVIDIA L40 (48GB)
- **RAM**: 117GB
- **CUDA**: 12.6
---
