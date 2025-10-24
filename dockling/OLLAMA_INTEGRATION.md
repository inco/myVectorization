# Ollama Integration Guide

## Обзор

Система теперь поддерживает Ollama для локальной генерации эмбеддингов и работы с LLM. Это позволяет использовать локальные модели без зависимости от внешних API.

## Настройка Ollama

### 1. Установка Ollama

```bash
# Windows
winget install Ollama.Ollama

# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Запуск Ollama сервера

```bash
ollama serve
```

### 3. Установка моделей

```bash
# Модель для эмбеддингов
ollama pull nomic-embed-text:latest

# Модель для генерации текста
ollama pull llama3:latest
```

## Конфигурация

### Переменные окружения

Создайте файл `.env` с настройками Ollama:

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://192.168.1.89:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest
OLLAMA_LLM_MODEL=llama3:latest
```

### Конфигурация в коде

```python
from src.config import get_config

config = get_config()
ollama_config = config.get_ollama_config()
print(f"Ollama URL: {ollama_config['base_url']}")
print(f"Embedding Model: {ollama_config['embedding_model']}")
print(f"LLM Model: {ollama_config['llm_model']}")
```

## Использование

### 1. Векторизация с Ollama

```bash
# Использование Ollama для эмбеддингов
python -m src.cli vectorize documents/ --vector-store qdrant --use-ollama --ollama-embedding-model nomic-embed-text:latest
```

### 2. Программное использование

```python
from src.library_vectorizer import LibraryVectorizer

# Создание векторизатора с Ollama
vectorizer = LibraryVectorizer(
    vector_store_type="qdrant",
    use_ollama=True,
    ollama_url="http://192.168.1.89:11434",
    ollama_embedding_model="nomic-embed-text:latest"
)

# Векторизация документов
result = vectorizer.vectorize_library("documents/", "my_collection")
```

### 3. RAG Pipeline с Ollama

```python
from src.ollama_integration import OllamaRAGPipeline

# Создание RAG пайплайна
rag = OllamaRAGPipeline(
    ollama_url="http://192.168.1.89:11434",
    embedding_model="nomic-embed-text:latest",
    llm_model="llama3:latest",
    vector_store_type="qdrant",
    collection_name="my_documents"
)

# Задавание вопросов
response = rag.ask_question("What is machine learning?", top_k=5)
print(response['answer'])
```

## Доступные модели

### Модели эмбеддингов

- `nomic-embed-text:latest` - Рекомендуемая модель для эмбеддингов
- `koill/sentence-transformers:all-minilm-l12-v2` - Альтернативная модель
- `mahonzhan/all-MiniLM-L6-v2:latest` - Легкая модель

### LLM модели

- `llama3:latest` - Рекомендуемая модель для генерации текста
- `qwen2.5:latest` - Альтернативная модель
- `mistral:latest` - Быстрая модель

## Преимущества Ollama

### ✅ Локальность
- Полный контроль над данными
- Нет зависимости от интернета
- Приватность данных

### ✅ Производительность
- Быстрая генерация эмбеддингов
- Низкая задержка
- Оптимизация для локального железа

### ✅ Гибкость
- Легкая замена моделей
- Настройка параметров
- Поддержка различных форматов

## Тестирование

### Базовый тест подключения

```bash
python test_ollama.py
```

### Тест векторизации

```bash
python -m src.cli vectorize documents/ml_overview.md --use-ollama --vector-store qdrant
```

## Устранение неполадок

### Проблема: Модель не найдена

```
Model nomic-embed-text not found. Available models: [...]
```

**Решение:**
```bash
ollama pull nomic-embed-text:latest
```

### Проблема: Сервер недоступен

```
[WinError 10061] No connection could be made because the target machine actively refused it
```

**Решение:**
1. Убедитесь, что Ollama сервер запущен: `ollama serve`
2. Проверьте URL в конфигурации
3. Проверьте сетевые настройки

### Проблема: Медленная генерация

**Решение:**
1. Используйте более легкие модели
2. Увеличьте batch_size в конфигурации
3. Оптимизируйте настройки Ollama

## Примеры использования

### Полный цикл RAG с Ollama

```python
#!/usr/bin/env python3

from src.library_vectorizer import LibraryVectorizer
from src.ollama_integration import OllamaRAGPipeline

# 1. Векторизация документов с Ollama
vectorizer = LibraryVectorizer(
    vector_store_type="qdrant",
    use_ollama=True,
    ollama_url="http://192.168.1.89:11434",
    ollama_embedding_model="nomic-embed-text:latest"
)

# Векторизация
result = vectorizer.vectorize_library("documents/", "my_library")
print(f"Vectorized {result['total_chunks']} chunks")

# 2. Создание RAG пайплайна
rag = OllamaRAGPipeline(
    ollama_url="http://192.168.1.89:11434",
    embedding_model="nomic-embed-text:latest",
    llm_model="llama3:latest",
    vector_store_type="qdrant",
    collection_name="my_library"
)

# 3. Задавание вопросов
questions = [
    "What is machine learning?",
    "How does neural networks work?",
    "What are the applications of AI?"
]

for question in questions:
    response = rag.ask_question(question, top_k=3)
    print(f"\nQ: {question}")
    print(f"A: {response['answer']}")
    print(f"Retrieved {response['num_documents']} documents")
```

## Заключение

Ollama интеграция предоставляет мощный инструмент для локальной работы с эмбеддингами и LLM. Это особенно полезно для:

- Приватных проектов
- Работы без интернета
- Контроля над данными
- Быстрой разработки

Система автоматически определяет доступные модели и использует их для генерации эмбеддингов и ответов на вопросы.

