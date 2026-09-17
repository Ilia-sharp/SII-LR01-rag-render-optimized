# ЛР1. RAG-ассистент по документам

Дисциплина «Системы искусственного интеллекта».

Сервис отвечает на вопросы по набору внутренних документов: находит релевантные фрагменты векторным поиском и собирает ответ **только из найденного текста**. Генеративной модели нет, поэтому выдумать факт сервис не может. Если подходящих фрагментов нет — возвращается `«Не нашёл информации по вашему вопросу»`.

---

## 1. Быстрый старт (≤ 5 минут)

Требуется Python 3.10+. Для запуска с компактной моделью (по умолчанию, для Render Free) интернет **не нужен** — модель уже лежит в `desktop/bundle/` внутри репозитория.

```bash
git clone <URL этого репозитория>
cd SII-LR01-rag

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt   # ~30 секунд, только lightweight-зависимости

uvicorn app.main:app --port 8000  # сервис на http://localhost:8000
```

Проверка в другом терминале:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"За сколько дней нужно подать заявление на отпуск?"}'
```

Готовый сценарий проверки: `bash scripts/smoke_test.sh`
Ответ без запуска сервера: `python scripts/ask.py "Как оплачивается переработка?"`
Интерактивная документация API: <http://localhost:8000/docs>

---

## 1.1 Деплой на Render.com Free Tier (512 МБ ОЗУ, 0.1 CPU)

Проект оптимизирован под бесплатный тариф Render: в runtime не используется
PyTorch и ChromaDB. Вместо них — ONNX Runtime + numpy. Пиковое потребление
памяти — **~180 МБ** (запас >330 МБ от лимита 512 МБ).

### Что изменилось

| Было (heavy) | Стало (light, по умолчанию) |
| --- | --- |
| sentence-transformers + torch (~1.5 ГБ RAM) | onnxruntime + tokenizers (~120 МБ RAM) |
| ChromaDB (sqlite + hnswlib, ~150 МБ) | numpy-массив (28 КБ, 18 чанков) |
| `intfloat/multilingual-e5-small` (470 МБ модель) | `cointegrated/rubert-tiny2` (30 МБ ONNX int8) |
| `pymupdf` + `pypdf` (~80 МБ) | не нужны (индекс предсобран) |
| Скачивание модели при старте | bundle уже в репозитории |

### Как развернуть

1. **Запушьте репозиторий на GitHub/GitLab.** Файлы `desktop/bundle/model/model.onnx` (30 МБ) и `desktop/bundle/index.npz` (21 КБ) обязательно должны попасть в коммит — `.gitignore` уже настроен не игнорировать их.
2. На [render.com](https://render.com) создайте **Web Service → New → Web Service → подключите репозиторий**.
3. Render автоматически прочитает `render.yaml` и применит конфигурацию:
   - **Plan:** Free
   - **Runtime:** Python 3.11
   - **Build:** `pip install -r requirements.txt` (~30 секунд, ~150 МБ зависимостей)
   - **Start:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1 --no-access-log`
   - **Health check:** `GET /health`
4. Через 1-2 минуты сервис будет доступен по адресу вида `https://sii-lr01-rag.onrender.com`.
5. Проверьте: `curl https://sii-lr01-rag.onrender.com/health` → `{"status":"ok"}`.

### Переменные окружения на Render

| Ключ | Значение | Зачем |
| --- | --- | --- |
| `RAG_BACKEND` | `light` | использовать ONNX-бэкенд вместо тяжёлого |
| `RAG_MODEL` | `cointegrated/rubert-tiny2` | для отображения в `/info` |
| `RAG_SCORE_THRESHOLD` | `0.40` | откалиброван для rubert-tiny2 |
| `OMP_NUM_THREADS` | `1` | не «душить» 0.1 CPU |
| `ORT_NUM_THREADS` | `1` | то же для ONNX Runtime |
| `HF_HUB_OFFLINE` | `1` | не ходить на HuggingFace (модель уже в bundle) |

Все значения уже прописаны в `render.yaml`, дополнительно настраивать ничего не нужно.

### Локально heavy-режим (если нужен torch + chromadb)

```bash
pip install -r requirements-dev.txt   # потянет torch, chromadb, pymupdf
python scripts/build_index.py         # построит .chroma/ индекс
RAG_BACKEND=heavy RAG_MODEL=intfloat/multilingual-e5-small \
  uvicorn app.main:app --port 8000
```

---

## 2. Архитектура (кратко)

В проекте два бэкенда энкодера и хранилища (см. `app/backends.py`):

1. **Light (по умолчанию, для Render Free)** — `desktop/bundle/` содержит ONNX-модель `cointegrated/rubert-tiny2` (30 МБ int8) и предсобранный индекс `index.npz` (numpy). При старте загружается за ~1 секунду.
2. **Heavy (для локальной разработки)** — `sentence-transformers` + `chromadb`, модель `intfloat/multilingual-e5-small`. Логика индексации и поиска та же, что в лабораторной — `scripts/build_index.py`.

Логика нарезки и сборки ответа одна для обоих бэкендов:

1. **Индексация** (только для heavy) — `scripts/build_index.py`: читает `data/`, режет модулем `app/chunker.py`, векторизует и складывает в ChromaDB. Для light-бэкенда индекс уже собран в `desktop/bundle/index.npz` скриптом `scripts/build_onnx_bundle_tiny.py`.
2. **Единая нарезка** — `app/chunker.py` вызывается и при индексации, и при обработке запроса.
3. **Единый энкодер** — одна и та же модель для документов и вопросов. Имя модели пишется в метаданные индекса; при старте API сверяет его с текущей конфигурацией.
4. **Поиск и фильтрация** — `app/rag.py`: top-k чанков по косинусной схожести, отсечение по порогу + лексическая проверка.
5. **Ответ** — экстрактивный: из топ-2 чанков выбираются 3 предложения, наиболее близкие к вопросу.

Поток данных:
- **light:** `desktop/bundle/index.npz` → `LightVectorStore` → `rag.search` → `rag.build_answer` → `POST /ask`
- **heavy:** `data/*` → `loader` → `chunker` → `embedder` → `ChromaDB` → `rag.search` → `rag.build_answer` → `POST /ask`

### Скрипты сборки bundle

| Скрипт | Что делает | Когда запускать |
| --- | --- | --- |
| `scripts/build_onnx_bundle_tiny.py` | Экспортирует `rubert-tiny2` в ONNX int8 (30 МБ), строит numpy-индекс | только при обновлении корпуса `data/` или смене модели |
| `scripts/build_onnx_bundle.py` | То же для `multilingual-e5-small` (119 МБ, тяжёлый вариант) | если нужна максимальная точность и есть RAM |
| `scripts/build_index.py` | Строит индекс в ChromaDB (heavy-бэкенд) | только для локальной разработки без ONNX |
| `scripts/calibrate.py` | Калибрует порог схожести | при смене модели или корпуса |

---

## 3. Корпус документов

**Источник:** корпус подготовлен специально для лабораторной работы — это синтетический набор внутренних регламентов вымышленной компании ООО «Ромашка». Персональных данных и защищённых авторским правом текстов в нём нет, документы написаны по типовой структуре корпоративных HR/IT-политик и FAQ.

**Тема:** внутренние правила и справочная информация для сотрудников. 9 документов в формате Markdown, около 2900 слов, 18 чанков в индексе.

| Файл | Содержание |
| --- | --- |
| `hr_policy.md` | Отпуска, больничные, отгулы, увольнение |
| `onboarding.md` | Оформление, первый день, испытательный срок, наставник |
| `remote_work.md` | Удалённый и гибридный формат, оборудование, бронирование мест |
| `it_support_faq.md` | Заявки в поддержку, пароли, VPN, доступы, замена техники |
| `business_trips.md` | Командировки, лимиты, суточные, авансовый отчёт |
| `security_policy.md` | Пароли и 2FA, классификация данных, инциденты, фишинг |
| `benefits.md` | ДМС, обучение, спорт, питание, материальная помощь |
| `work_schedule.md` | График, учёт времени, переработки, дежурства |
| `payroll_and_review.md` | Зарплата, премии, оценка результативности, справки |

Форматы `.txt` и `.pdf` тоже поддерживаются: положите файлы в `data/` и перестройте индекс. PDF читается через `pymupdf` (запасной вариант — `pypdf`).

---

## 4. API

### `GET /health`

```json
{"status": "ok"}
```

### `POST /ask`

Запрос:

```json
{"question": "За сколько дней нужно подать заявление на отпуск?"}
```

Ответ:

```json
{
  "answer": "Согласно документам: Заявление на отпуск подаётся не позднее чем за 14 рабочих дней до даты начала отпуска через портал HR-Self-Service. Отпускные перечисляются не позднее чем за 3 календарных дня до начала отпуска.",
  "sources": [
    {"file": "hr_policy.md", "chunk_id": 0, "score": 0.8931},
    {"file": "work_schedule.md", "chunk_id": 1, "score": 0.8415}
  ],
  "model_version": "intfloat/multilingual-e5-small"
}
```

Вопрос не по теме корпуса:

```json
{
  "answer": "Не нашёл информации по вашему вопросу",
  "sources": [],
  "model_version": "intfloat/multilingual-e5-small"
}
```

`chunk_id` — порядковый номер фрагмента внутри файла, начиная с 0. `score` — косинусная схожесть в диапазоне 0…1 (чем больше, тем ближе).

Коды ответов: `200` — успех, `422` — пустой или слишком длинный вопрос, `503` — индекс не построен или построен другой моделью, `500` — внутренняя ошибка.

### `GET /`

Веб-интерфейс демонстрации: поле для вопроса, готовые примеры, ответ с источниками и кнопка «Показать ответ API (JSON)». Открывается в браузере по адресу <http://localhost:8000>.

### `GET /info`

Служебная информация: статус индекса, модель, количество файлов и чанков.

### Проверка из PowerShell

В PowerShell `curl` — это псевдоним `Invoke-WebRequest`, и команда из раздела 1 там работает иначе. Используйте `curl.exe` либо родную команду:

```powershell
Invoke-RestMethod -Uri http://localhost:8000/ask -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body ([Text.Encoding]::UTF8.GetBytes('{"question":"За сколько дней нужно подать заявление на отпуск?"}'))
```

Кодировку приходится задавать явно, иначе кириллица в теле запроса уедет. Проще всего проверять через веб-интерфейс на <http://localhost:8000> или через Swagger на <http://localhost:8000/docs>.

---

## 5. Логирование

В лог пишутся только метрики, без текста вопроса и ответа:

```
INFO rag.api | ask q_hash=6f2a1c9b0d q_len=48 chunks_found=2 latency_ms=41.7 answered=True
```

`q_hash` — первые 10 символов SHA-256 от вопроса (позволяет отличать запросы, не раскрывая содержимое), `q_len` — длина вопроса, `chunks_found` — число фрагментов выше порога, `latency_ms` — время обработки.

---

## 6. Конфигурация

Все параметры задаются переменными окружения, значения по умолчанию — в `app/config.py` (см. `.env.example`).

| Переменная | По умолчанию | Назначение |
| --- | --- | --- |
| `RAG_BACKEND` | `light` | `light` = ONNX+numpy (для Render), `heavy` = torch+chromadb |
| `RAG_MODEL` | `cointegrated/rubert-tiny2` | энкодер (один и тот же для индексации и запросов) |
| `RAG_CHUNK_WORDS` | `220` | размер чанка в словах (≈ 300-450 токенов) |
| `RAG_CHUNK_OVERLAP` | `0.15` | перекрытие чанков (15 %) |
| `RAG_TOP_K` | `3` | сколько фрагментов возвращать в `sources` |
| `RAG_SCORE_THRESHOLD` | `0.35` для rubert-tiny2 / `0.80` для E5 | порог схожести для ответа «не нашёл» |
| `RAG_LEXICAL_GUARD` | `1` | дополнительная проверка совпадения слов вопроса и чанка |
| `RAG_DATA_DIR` / `RAG_CHROMA_DIR` | `data` / `.chroma` | пути (только для heavy-бэкенда) |
| `RAG_BUNDLE_DIR` | `desktop/bundle` | путь к предсобранному ONNX-bundle (light-бэкенд) |
| `OMP_NUM_THREADS` / `ORT_NUM_THREADS` | `1` | лимит потоков ONNX Runtime (важно для 0.1 CPU) |
| `HF_HUB_OFFLINE` | `1` | не ходить на HuggingFace (модель уже в bundle) |

**Почему rubert-tiny2 на Render.** Чанк в 220 слов ≈ 300-450 токенов, нужен энкодер с окном 512 токенов. У `rubert-tiny2`:
- контекст 512 токенов ✓
- специально обучена на русском (качество лучше, чем у многоязычной e5-small)
- размер 30 МБ int8 ONNX vs 119 МБ у e5-small
- пиковое потребление RAM ~180 МБ vs ~550 МБ — критично для лимита 512 МБ

Служебные префиксы `query:` / `passage:`, которые требует семейство E5, для rubert-tiny2 НЕ нужны (это BERT, а не E5). В `app/backends.py` префиксы автоматически включаются по флагу `e5_prefixes` в метаданных bundle.

**Про порог.** У E5 косинусные схожести «сжаты» (нерелевантная пара даёт ≈ 0.75), поэтому порог высокий — 0.80. У rubert-tiny2 шкала растянута, нерелевантная пара ≈ 0.33, релевантная ≈ 0.40 — порог 0.35 выбран калибровкой. Лексический guard (`RAG_LEXICAL_GUARD=1`) дополнительно отсекает оставшиеся пограничные случаи. Откалибровать порог на своём корпусу:

```bash
python scripts/calibrate.py
```

Скрипт прогоняет вопросы по корпусу и вне корпуса, печатает score и предлагает значение порога.

---

## 7. Тесты

```bash
pip install -r requirements-dev.txt
pytest -q
```

Тесты нарезки и фильтрации (`tests/test_chunker.py`, `tests/test_rag_guard.py`) запускаются без модели и индекса; `tests/test_api_health.py` проверяет `/health` и валидацию запроса.

---

## 7.1 Частые проблемы на Windows

**`source venv/bin/activate` не работает.** Это команда для Linux/macOS. В PowerShell: `.\venv\Scripts\Activate.ps1`, в cmd: `venv\Scripts\activate.bat`.

**«running scripts is disabled on this system».** Разрешите запуск скриптов для текущего пользователя: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.

**`Python was not found`.** Сработал ярлык Microsoft Store. Установите Python с python.org с галочкой «Add python.exe to PATH» либо отключите псевдонимы: Параметры → Приложения → Дополнительные параметры приложений → Псевдонимы выполнения приложений.

**Не клонируйте проект в `C:\Windows\System32`.** Эта папка системная, на неё действуют права администратора и защита Windows: индекс и виртуальное окружение могут не записаться. Работайте в своей папке, например `C:\Users\<имя>\Projects\SII-LR01-rag`.

**Показ без интернета.** После первого запуска модель лежит в кеше `C:\Users\<имя>\.cache\huggingface`, но библиотека всё равно ходит на huggingface.co проверять обновления и без сети может подвисать на старте. Перед демонстрацией задайте офлайн-режим:

```powershell
$env:HF_HUB_OFFLINE = "1"
uvicorn app.main:app --port 8000
```

## 8. Структура репозитория

```
SII-LR01-rag/
├── app/
│   ├── config.py       # пути, модель, параметры чанкинга и порогов
│   ├── chunker.py      # ЕДИНЫЙ модуль нарезки (индексация + запрос)
│   ├── loader.py       # чтение .md/.txt/.pdf из data/
│   ├── embedder.py     # sentence-transformers, префиксы E5 (heavy-бэкенд)
│   ├── store.py        # ChromaDB + метаданные индекса (heavy-бэкенд)
│   ├── backends.py     # переключение light/heavy-бэкендов
│   ├── rag.py          # поиск, фильтрация, сборка ответа
│   ├── main.py         # FastAPI: GET /, POST /ask, GET /health
│   └── ui.html         # веб-интерфейс (общий с desktop-сборкой)
├── desktop/
│   ├── onnx_encoder.py # ONNX-энкодер + numpy-индекс (light-бэкенд)
│   ├── demo_app.py     # автономная exe-сборка
│   ├── prepare_bundle.py
│   └── bundle/         # ★ предсобранный ONNX-bundle (коммитится в репозиторий!)
│       ├── model/model.onnx       # 30 МБ, rubert-tiny2 int8
│       ├── model/tokenizer.json   # 2.4 МБ
│       ├── index.npz              # 21 КБ, векторы корпуса
│       └── index.json             # 45 КБ, метаданные
├── scripts/
│   ├── build_index.py             # индексация для heavy-бэкенда
│   ├── build_onnx_bundle.py      # сборка ONNX-bundle для e5-small (тяжёлый)
│   ├── build_onnx_bundle_tiny.py # сборка ONNX-bundle для rubert-tiny2 (по умолчанию)
│   ├── ask.py                     # ответ из консоли без сервера
│   ├── calibrate.py              # калибровка порога схожести
│   └── smoke_test.sh             # проверка запущенного сервиса через curl
├── data/               # корпус: 9 документов
├── tests/
├── requirements.txt        # lightweight-зависимости (для Render)
├── requirements-dev.txt    # + heavy-зависимости (torch, chromadb) для локалки
├── render.yaml             # ★ конфигурация Render.com
├── Procfile                # ★ alternative-конфиг для Heroku/Render
├── runtime.txt             # ★ Python 3.11
├── .python-version         # ★ Python 3.11
├── .env.example
├── Makefile
└── README.md
```

Короткие команды: `make install`, `make index`, `make run`, `make test`, `make smoke`.

### Автономная демонстрация (exe)

В папке `desktop/` лежит сборка демо в один исполняемый файл: запускается без Python, интернета и установки, модель и индекс внутри. Собирается один раз двойным кликом по `desktop\build_exe.bat`, результат — `dist\RAG-Demo.exe`. Подробности и сценарий показа — в `desktop/README.md`.

---

## 9. Соответствие требованиям ЛР

| # | Требование | Где реализовано |
| --- | --- | --- |
| 1 | Скрипт индексации `python scripts/build_index.py` | `scripts/build_index.py` |
| 2 | `POST /ask` с телом `{"question": "..."}` | `app/main.py` |
| 3 | Ответ с `answer`, `sources` (файл + `chunk_id` + `score`), `model_version` | `app/main.py`, `app/rag.py` |
| 4 | `GET /health` → `{"status": "ok"}` | `app/main.py` |
| 5 | Логи: длина/хеш вопроса, `latency_ms`, число чанков, без текстов | `app/main.py` |
| 6 | README с установкой и запуском за ≤ 5 минут | раздел 1 |
| — | Чанки 256-512 токенов, перекрытие 10-20 % | `app/config.py`, `app/chunker.py` |
| — | Единый модуль нарезки для индексации и запроса | `app/chunker.py` |
| — | Один энкодер для индексации и запросов, сверка по метаданным индекса | `app/embedder.py`, `app/main.py` |
| — | «Не нашёл информации по вашему вопросу» при низкой схожести | `app/rag.py` |
| — | Относительные пути, `.gitignore`, без выдумывания ответа | `app/config.py`, `.gitignore`, экстрактивная сборка |

---

## 10. Чек-лист перед сдачей

- [ ] Репозиторий `SII-LR01-<фамилия>` создан на GitVerse, добавлен collaborator **MaxxxVS**
- [ ] `pip install -r requirements.txt` отрабатывает без ошибок
- [ ] `python scripts/build_index.py` строит индекс из `data/`
- [ ] `curl -X POST .../ask` возвращает `answer`, `sources`, `model_version`
- [ ] Вопрос по несуществующей теме возвращает «Не нашёл информации по вашему вопросу»
- [ ] README содержит инструкцию по запуску
- [ ] Проставлен и отправлен тег `v1.0`

Публикация:

```bash
git init
git add .
git commit -m "ЛР1: RAG-ассистент по документам"
git remote add origin https://gitverse.ru/<логин>/SII-LR01-<фамилия>.git
git push -u origin master
git tag v1.0
git push origin v1.0
```
