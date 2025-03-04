# Projekt: Chat-bot for psychological counselling

## Zbiór danych
- **Liczba tekstów**: 3,512
- **Format**: JSON
- **Anotacje**: Każda próbka zawiera dwie kolumny tekstowe **"Context"** i **"Response"**
- **Link do zbioru danych**: [Mental Health Counseling Conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations)

### Podział na zbiór train, val i test:
Nabor danych ładowany jest w całości i wykorzystywany do trenowania modelu

### Przetwarzanie wstępne:
Funkcja `preprocess_function` łączy  **Context** i **Response** w jeden ciąg tekstowy, dodając odpowiednie metki, takie jak: `<s>[INST] {context} [/INST] {response} </s>`

---
## Model i architektura
- **Architektura**: Transformer
- **Użyty model**: 
  - **LLaMA-2-7b-chat-hf**
  - **LLaMA-3.2-1B-Instruct**
  - **LLaMA-68m**

### Modyfikacje i dostrojenie modelu:
1. **LoRA**: Technika adaptacji dużych modeli do specyficznych zadań при minimalnym koszcie obliczeniowym
2. **QLoRA**: Kwantowanie 4-bitowe, które zmniejsza rozmiar modelu, oszczędza pamięć i przyspiesza obliczenia

### Inspiracja:
- [Fine-tuning LLaMA-2 using LoRA and QLoRA - A Comprehensive Guide](https://medium.com/@harsh.vardhan7695/fine-tuning-llama-2-using-lora-and-qlora-a-comprehensive-guide-fd2260f0aa5f)
---
## Metodologia

### Preprocessing:
- **AutoTokenizer** dzieli tekst na tokeny i zamienia je na liczby, a następnie przygotowuje prompt w odpowiednim formacie:
  `"<s>[INST] {prompt} [/INST]"`

### Trening:
- Model **LLaMA** jest dostosowywany do danych przy użyciu **fine-tuning** z technikami **LoRA** i **QLoRA**

### Ewaluacja:
- Ocena jakości tekstu odbywa się przy pomocy metryk:
  - **BLEU**
  - **Perplexity**
  - **VADER Empathy Score**
  - **Relevance Score**

### Modele i podejścia (prompting):
- **Użyty model**: LLaMA-2-7b-chat-hf, LLaMA-3.2-1B-Instruct, LLaMA-68m
- **Instruction-based prompting**: Model reaguje na instrukcje w postaci tekstu, generując odpowiedzi na podstawie przekazanych promptów
---
## Miary jakości modelu:
1. **BLEU**: Mierzy jakość generowanego tekstu w porównaniu do referencyjnych odpowiedzi
   - **Kryteria akceptacji**: BLEU > 0.3
2. **Perplexity**: Mierzy, jak dobrze model przewiduje następne słowo w sekwencji
   - **Kryteria akceptacji**: Perplexity < 0.5
3. **VADER Empathy Score**: Ocena emocjonalnego tonu wygenerowanego tekstu
   - **Kryteria akceptacji**: Empathy score > 0.5
4. **Dialog Quality Metrics**: Ocena długości odpowiedzi (length) i liczby unikalnych słów (unique words), 
co pomaga określić szczegółowość i różnorodność odpowiedzi
   - **Kryteria akceptacji**: 
   - **Length**: Odpowiedzi o odpowiedniej długości
   - **Unique words**: Wyższa liczba unikalnych słów = większa różnorodność odpowiedzi
5. **Relevance Score**: Ocena, jak silnie wygenerowany tekst odpowiada na zapytanie
   - **Kryteria akceptacji**: Relevance score > 0.7

### Wyniki:
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
## Dodatkowe biblioteki:
- **Transformers**
- **Peft**
- **SpaCy**
- **NLTK**
- **Torch**

## Środowisko sprzętowe i programowe:
- **GPU**: NVIDIA L40 (48GB)
- **RAM**: 117GB
- **CUDA**: 12.6
---
