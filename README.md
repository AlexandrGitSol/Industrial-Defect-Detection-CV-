# Обнаружение дефектов на производстве (CV)

Детекция аномалий на промышленных изделиях без учителя с помощью **PaDiM** на датасете **MVTec AD**. Система учится только на **нормальных** образцах и помечает как дефект всё, что от них отличается — размеченные примеры брака не нужны.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Anomalib](https://img.shields.io/badge/Anomalib-2.x-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal)
![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey)

---

## О проекте

Проект реализует полный пайплайн для обнаружения поверхностных дефектов на manufactured объектах (бутылки, металлические гайки и т.д.) методом **anomaly detection**. В отличие от supervised-подходов, которым нужны тысячи размеченных примеров брака, этот метод изучает распределение «нормального» внешнего вида и замечает любое отклонение от него.

**Ключевая идея:** PaDiM извлекает признаки из предобученной CNN (ResNet18) для каждого патча изображения, моделирует их распределение как многомерные гауссианы и использует расстояние Махаланобиса для оценки аномальности на тесте.

### Почему это важно

В реальном производстве дефекты редки и разнообразны. Вы не можете собрать датасет со всеми возможными типами брака. Anomaly detection решает это, изучая **только то, как выглядит «норма»** — и помечая всё остальное как отклонение.

---

## Возможности

- **Обучение без учителя** — нужны только нормальные изображения
- **Попиксельная локализация** — выдаёт тепловые карты, показывающие, *где* дефект
- **Работает на малых данных** — достаточно ~200 нормальных изображений на категорию
- **Быстрый инференс** — без градиентного спуска; «обучение» — это построение гауссовой статистики
- **HTTP API** — FastAPI-сервер для интеграции с другими системами (камеры, конвейеры, веб-приложения)
- **Базовые метрики** на MVTec AD (категория bottle):

| Метрика | Значение |
|---|---|
| Image AUROC | **0.9992** |
| Image F1 | **0.9841** |
| Pixel AUROC | **0.9810** |
| Pixel F1 | **0.6986** |

> *Примечание: Pixel F1 ниже, потому что PaDiM даёт «размытую» попиксельную локализацию — это известная особенность метода, а не баг.*

---

## Стек

| Слой | Технология |
|---|---|
| Язык | Python 3.11 |
| Anomaly Detection | Anomalib (Intel) |
| Deep Learning | PyTorch / PyTorch Lightning |
| Обработка изображений | OpenCV |
| API | FastAPI + Uvicorn |
| Датасет | MVTec AD (CC BY-NC-SA 4.0) |

---

## Структура проекта

    Industrial Defect Detection (CV)/
    ├── datasets/
    │   └── MVTecAD/                  # Датасет MVTec AD (bottle, cable и т.д.)
    ├── results/
    │   └── Padim/MVTecAD/bottle/     # Чекпоинты, метрики, тепловые карты
    ├── src/
    │   ├── inference.py              # Загрузка чекпоинта и предсказание
    │   └── api.py                    # (опционально) FastAPI-сервер
    ├── server.py                     # HTTP-сервер (FastAPI)
    ├── client.py                     # Пример HTTP-клиента
    ├── train.py                      # Скрипт обучения
    ├── requirements.txt
    ├── .gitignore
    └── README.md

---

## Быстрый старт

### 1. Клонировать репозиторий

    git clone https://github.com/AlexandrGitSol/Industrial-Defect-Detection-CV-.git
    cd "Industrial Defect Detection (CV)"

### 2. Создать виртуальное окружение

    python3 -m venv venv
    source venv/bin/activate   # Linux / macOS
    # venv\Scripts\activate    # Windows

### 3. Установить зависимости

    pip install "anomalib[cpu]"
    pip install "fastapi[standard]"

### 4. Скачать веса модели

Обученные веса доступны на странице [Releases](https://github.com/AlexandrGitSol/Industrial-Defect-Detection-CV-/releases).

Скачайте `model.ckpt` и положите сюда:

    results/Padim/MVTecAD/bottle/v1/weights/lightning/model.ckpt

### 5. Скачать датасет (опционально, для обучения)

Скрипт скачивает MVTec AD автоматически при первом запуске. Если хотите обучить с нуля:

    python train.py

---

## Использование

### Вариант A: Инференс из Python

    from anomalib.engine import Engine
    from anomalib.models import Padim
    from anomalib.data import PredictDataset
    from pathlib import Path

    engine = Engine()
    model = Padim()

    dataset = PredictDataset(
        path=Path("datasets/MVTecAD/bottle/test/broken_large"),
        image_size=(256, 256),
    )

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path="results/Padim/MVTecAD/bottle/v1/weights/lightning/model.ckpt",
    )

    for pred in predictions:
        print(f"Путь: {pred.image_path}")
        print(f"Метка (0=норма, 1=дефект): {pred.pred_label}")
        print(f"Оценка аномальности: {pred.pred_score.item():.4f}")

### Вариант B: Запустить HTTP API

    uvicorn server:app --host 0.0.0.0 --port 8000

Откройте `http://localhost:8000/docs` — FastAPI автоматически сгенерирует интерактивную документацию Swagger.

### Вариант C: Отправить картинку через HTTP-клиент

    python client.py datasets/MVTecAD/bottle/test/broken_large/000.png

Или через `curl`:

    curl -X POST http://localhost:8000/predict -F "file=@datasets/MVTecAD/bottle/test/broken_large/000.png"

Ответ:

    {
      "label": "defect",
      "score": 0.94
    }

---

## Как это работает

1. **Обучение (без учителя):**
   - PaDiM прогоняет нормальные изображения через предобученный ResNet18.
   - Для каждого пространственного патча собираются признаки из нескольких слоёв.
   - Для каждой позиции патча строится многомерное гауссово распределение (среднее + ковариация).

2. **Инференс:**
   - Тестовое изображение проходит через тот же backbone.
   - Для каждого патча вычисляется расстояние Махаланобиса до выученного гауссиана.
   - Большое расстояние — аномалия. Карта аномалий апскейлится до попиксельного уровня.

3. **Почему работает на малых данных:**
   - Backbone **предобучен** на миллионах изображений (ImageNet). Он уже «умеет» извлекать осмысленные признаки.
   - PaDiM нужно выучить только *распределение* этих признаков на ваших нормальных данных — это статистический шаг, а не градиентный спуск.

---

## Ограничения

- **Чувствителен к позе объекта** — PaDiM моделирует распределение патчей в фиксированных пространственных позициях. При значительном сдвиге или повороте качество падает (~5% AUROC на невыровненных данных).
- **Грубая попиксельная локализация** — тепловые карты могут быть шире реального дефекта.
- **Не real-time на CPU** — для продакшена рекомендуется GPU.

---

## Планы

- [ ] Сравнить с PatchCore и EfficientAD
- [ ] Добавить маскирование фона для устойчивости к сдвигам
- [ ] Экспорт в ONNX для C++ деплоя
- [ ] Завернуть в Docker
- [ ] Добавить unit-тесты для пайплайна инференса

---

## Источники

- Defard et al. (2021). *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization.*
- Bergmann et al. (2019). *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.*
- [Документация Anomalib](https://anomalib.readthedocs.io/)
- [Документация FastAPI](https://fastapi.tiangolo.com/)

---

## Лицензия

Проект использует датасет **MVTec AD**, который распространяется под лицензией **CC BY-NC-SA 4.0** (только некоммерческое использование). Код предоставлен в образовательных и портфолио-целях.

---

## Автор

**Alexandr** — [GitHub](https://github.com/AlexandrGitSol)

Если есть вопросы или предложения — открывайте issue.
