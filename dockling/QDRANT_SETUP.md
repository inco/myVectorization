# Library Vectorization RAG System - Qdrant Setup

## ✅ Успешно настроено!

Ваша система векторизации библиотеки в RAG с Docling успешно настроена и работает с Qdrant.

### 🔧 Конфигурация Qdrant

- **Host**: 192.168.1.13
- **Port**: 6333
- **API Key**: 4c77aa13-30ff-41e0-ac87-ab1e0c791d2e
- **Collection**: market_documents

### 📊 Результаты тестирования

✅ **Векторизация**: 26 чанков успешно обработано и сохранено в Qdrant  
✅ **RAG Query**: Система корректно отвечает на вопросы  
✅ **Подключение**: Стабильное соединение с Qdrant сервером  

## 🚀 Быстрый старт

### 1. Использование готового скрипта

```bash
python qdrant_quickstart.py
```

Этот скрипт предоставляет интерактивное меню для:
- Векторизации документов
- Тестирования запросов
- Просмотра статистики
- Интерактивного Q&A

### 2. Прямые команды CLI

#### Векторизация документов
```bash
# Установить переменные окружения
$env:QDRANT_HOST="192.168.1.13"
$env:QDRANT_PORT="6333"
$env:QDRANT_API_KEY="4c77aa13-30ff-41e0-ac87-ab1e0c791d2e"
$env:QDRANT_COLLECTION="market_documents"

# Векторизовать документы
python -m src.cli vectorize ./documents --vector-store qdrant --collection-name market_documents
```

#### Тестирование запросов
```bash
# Одиночный запрос
python -m src.cli query "What is machine learning?" --vector-store qdrant --collection-name market_documents

# Интерактивный режим
python -m src.cli ask --vector-store qdrant --collection-name market_documents

# Поиск документов
python -m src.cli search --vector-store qdrant --collection-name market_documents

# Статистика коллекции
python -m src.cli stats --vector-store qdrant --collection-name market_documents
```

## 📁 Структура проекта

```
myLibraryScan/
├── src/
│   ├── __init__.py
│   ├── library_vectorizer.py    # Основной модуль векторизации
│   ├── rag_pipeline.py          # RAG pipeline
│   ├── cli.py                   # CLI интерфейс
│   └── config.py                # Конфигурация
├── documents/                   # Ваши документы
│   └── ml_overview.md
├── examples/                    # Примеры использования
├── tests/                       # Тесты
├── requirements.txt             # Зависимости
├── qdrant_quickstart.py        # Скрипт быстрого старта
├── README.md                    # Основная документация
└── USAGE.md                     # Инструкции по использованию
```

## 🔧 Поддерживаемые форматы документов

- **PDF** (.pdf) - с продвинутым парсингом и пониманием структуры
- **Microsoft Word** (.docx, .doc)
- **Plain Text** (.txt)
- **Markdown** (.md)

## 🧠 Стратегии chunking

- **Hierarchical** (по умолчанию) - сохраняет структуру документа
- **Hybrid** - комбинирует семантическое и структурное разбиение

## 🔍 Модели эмбеддингов

- **BAAI/bge-small-en-v1.5** (384 dim) - по умолчанию, быстрая
- **BAAI/bge-base-en-v1.5** (768 dim) - сбалансированная
- **BAAI/bge-large-en-v1.5** (1024 dim) - высокое качество

## 📈 Примеры использования

### Python API

```python
from src.library_vectorizer import LibraryVectorizer
from src.rag_pipeline import SimpleRAG

# Векторизация
vectorizer = LibraryVectorizer(
    vector_store_type="qdrant",
    embedding_model="BAAI/bge-small-en-v1.5",
    chunking_strategy="hierarchical"
)

result = vectorizer.vectorize_library("./documents", "market_documents")

# RAG система
rag = SimpleRAG(vector_store_type="qdrant", collection_name="market_documents")

# Задать вопрос
answer = rag.ask("What is machine learning?")
print(answer)

# Поиск документов
docs = rag.search("machine learning", top_k=5)
for doc in docs:
    print(f"Source: {doc['metadata']['source_file']}")
    print(f"Score: {doc['score']}")
    print(f"Content: {doc['text'][:100]}...")
```

## 🔄 Добавление новых документов

1. Поместите новые документы в папку `documents/`
2. Запустите векторизацию:
   ```bash
   python -m src.cli vectorize ./documents --vector-store qdrant --collection-name market_documents
   ```
3. Документы будут добавлены в существующую коллекцию

## 🛠️ Настройка и конфигурация

### Переменные окружения

```bash
# Qdrant настройки
QDRANT_HOST=192.168.1.13
QDRANT_PORT=6333
QDRANT_API_KEY=4c77aa13-30ff-41e0-ac87-ab1e0c791d2e
QDRANT_COLLECTION=market_documents

# Модель эмбеддингов
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Стратегия chunking
CHUNKING_STRATEGY=hierarchical

# RAG настройки
RAG_TOP_K=5
```

### Изменение модели эмбеддингов

```bash
python -m src.cli vectorize ./documents \
  --vector-store qdrant \
  --collection-name market_documents \
  --embedding-model "BAAI/bge-large-en-v1.5"
```

## 🐛 Устранение неполадок

### Проблемы с подключением к Qdrant

1. **SSL ошибки**: Система автоматически использует HTTP вместо HTTPS
2. **Таймауты**: Проверьте доступность сервера 192.168.1.13:6333
3. **API ключ**: Убедитесь, что ключ корректный

### Проблемы с памятью

- Используйте меньшую модель эмбеддингов: `BAAI/bge-small-en-v1.5`
- Обрабатывайте документы пакетами
- Увеличьте размер swap файла

### Проблемы с производительностью

- Используйте GPU для эмбеддингов (если доступен)
- Настройте batch_size в конфигурации
- Используйте более быструю модель эмбеддингов

## 📞 Поддержка

- **Документация**: README.md и USAGE.md
- **Примеры**: examples/basic_examples.py
- **Тесты**: tests/test_system.py
- **Логи**: Включите verbose режим: `--verbose`

## 🎯 Следующие шаги

1. **Добавьте больше документов** в папку `documents/`
2. **Настройте LLM интеграцию** для более качественных ответов
3. **Экспериментируйте с разными моделями** эмбеддингов
4. **Настройте мониторинг** производительности
5. **Создайте веб-интерфейс** для удобного использования

---

**🎉 Поздравляем! Ваша система RAG с Docling и Qdrant готова к использованию!**
