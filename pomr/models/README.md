# Папка для моделей VoiceToTaskApp

## 📦 Структура папки

```
models/
├── whisper/              # Локальные модели Whisper для транскрибации
│   └── openai-whisper-small/  # или medium/large
├── llama3.2/            # Локальные LLM модели для анализа задач
│   └── Meta-Llama-3.2-3B-Instruct/
└── mistral/             # Альтернативные LLM модели
    └── Mistral-7B-Instruct-v0.3/
```

## 🚀 Загрузка моделей

### Способ 1: Через huggingface-cli (рекомендуется)

```bash
# Установка huggingface-cli
pip install huggingface-hub

# Авторизация на Hugging Face
huggingface-cli login

# Загрузка Whisper модели
huggingface-cli download openai/whisper-small \
    --local-dir models/whisper/openai-whisper-small \
    --include "*.safetensors" "*.json" "*.txt"

# Загрузка Llama 3.2 3B (рекомендуется)
huggingface-cli download meta-llama/Meta-Llama-3.2-3B-Instruct \
    --local-dir models/llama3.2/Meta-Llama-3.2-3B-Instruct \
    --include "*.safetensors" "*.json" "*.txt" "tokenizer*" "config.json"

# Альтернатива: Mistral 7B
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 \
    --local-dir models/mistral/Mistral-7B-Instruct-v0.3 \
    --include "*.safetensors" "*.json" "*.txt" "tokenizer*" "config.json"
```

### Способ 2: Через Python скрипт

Создайте файл `download_models.py`:

```python
from huggingface_hub import hf_hub_download, HfApi

api = HfApi()

# Загрузка Whisper
whisper_model = "openai/whisper-small"
api.hf_hub_download(
    repo_id=whisper_model,
    filename="model.safetensors",
    local_dir="models/whisper",
    local_dir_use_symlinks=False
)

# Загрузка Llama 3.2
llama_model = "meta-llama/Meta-Llama-3.2-3B-Instruct"
api.hf_hub_download(
    repo_id=llama_model,
    filename="model.safetensors",
    local_dir="models/llama3.2",
    local_dir_use_symlinks=False
)
```

Запуск:
```bash
python download_models.py
```

### Способ 3: Через transformers (автоматическая загрузка)

Модели загружаются автоматически при первом запуске:

```python
from transformers import AutoModelForSpeechSeq2Seq, AutoModelForCausalLM

# При инициализации WhisperTranscriber
whisper = WhisperTranscriber(model_name="openai/whisper-small")
# Модель автоматически скачается в models/whisper/openai-whisper-small/

# При инициализации TaskAnalyzer
llm = TaskAnalyzer(model_name="meta-llama/Meta-Llama-3.2-3B-Instruct")
# Модель автоматически скачается в models/llama3.2/
```

## 📊 Размер моделей

| Модель | Размер | VRAM (мин.) | Рекомендация |
|--------|--------|-------------|---------------|
| Whisper-tiny | ~75 МБ | 1 ГБ | Быстро, базовое качество |
| Whisper-base | ~300 МБ | 2 ГБ | Баланс |
| **Whisper-small** | ~750 МБ | 3 ГБ | **Рекомендуется** |
| Whisper-medium | ~1.5 ГБ | 4 ГБ | Хорошее качество |
| Whisper-large-v3 | ~3 ГБ | 6 ГБ | Лучшее качество |
| Llama-3.2-1B | ~1 ГБ | 2 ГБ | Быстро, малый контекст |
| **Llama-3.2-3B** | ~2 ГБ | 4 ГБ | **Рекомендуется** |
| Mistral-7B | ~7 ГБ | 8 ГБ | Мощнее, больше память |

## 💻 Требования к железу

### Минимальные требования:
- **GPU:** NVIDIA с 4 ГБ VRAM (рекомендуется 6+ ГБ)
- **CPU:** 4+ ядер
- **RAM:** 8+ ГБ
- **Диск:** 10+ ГБ свободного места

### Для Whisper-small + Llama-3.2-3B:
- **GPU:** 4-6 ГБ VRAM
- **Время транскрибации:** ~3 сек/мин аудио
- **Время анализа:** ~5 сек/запрос

## 🔧 Использование квантованных моделей (меньше памяти)

Для экономии VRAM используйте квантованные модели:

```bash
# Квантованная Whisper-small (INT8)
huggingface-cli download openai/whisper-small \
    --local-dir models/whisper/whisper-small-int8 \
    --include "*.safetensors"

# Квантованный Llama-3.2-3B (4-bit)
huggingface-cli download meta-llama/Meta-Llama-3.2-3B-Instruct-Q4_K_M \
    --local-dir models/llama3.2/Meta-Llama-3.2-3B-Instruct-Q4_K_M \
    --include "*.safetensors" "*.json" "tokenizer*"
```

## 📝 Проверка загрузки моделей

```python
from src.whisper_transcriber import WhisperTranscriber
from src.llm_analyzer import TaskAnalyzer

# Проверка Whisper
whisper = WhisperTranscriber()
print(f"Whisper модель: {whisper.model_name}")
print(f"Устройство: {whisper.device}")

# Проверка LLM
llm = TaskAnalyzer()
print(f"LLM модель: {llm.model_name}")
print(f"Устройство: {llm.device}")
```

## 🔄 Обновление моделей

```bash
# Перезагрузка Whisper
huggingface-cli download openai/whisper-small \
    --local-dir models/whisper/openai-whisper-small \
    --force-download=True

# Перезагрузка LLM
huggingface-cli download meta-llama/Meta-Llama-3.2-3B-Instruct \
    --local-dir models/llama3.2/Meta-Llama-3.2-3B-Instruct \
    --force-download=True
```

## 🗑️ Очистка кэша моделей

```bash
# Очистка кэша Hugging Face
rm -rf ~/.cache/huggingface

# Или через Python
from huggingface_hub import scan_cache_dir
scan_cache_dir().clear_all()
```
