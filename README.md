# Павел Матевосов

### Data Scientist · Machine Learning Engineer

Строю ML-решения от анализа данных и валидации до API, контейнеризации и мониторинга. Основные направления: рекомендательные системы, табличный ML, обработка текстов и документов. Есть практический опыт PyTorch и CLIP, временных рядов и эконометрики.

**Data Science в экономике, РАНХиГС** · **Chemical Engineering, University of Manchester** · Английский **C2** · Москва

[Telegram](https://t.me/SapientiaVincit) · [Email](mailto:pmatevosov@yandex.ru) · [Карта проектов](projects/README.md)

## Избранные проекты

| Проект | Задача и инженерная часть | Результат |
|---|---|---|
| [Банковские рекомендации](https://github.com/matevosovp/practicum-sem4-praktika1) | Новые продукты на следующий месяц; CatBoost, временная валидация, MLflow Registry, API | Test MAP@3 **0,6527**, baseline **0,5753** на том же sampled holdout |
| [E-commerce recommender](https://github.com/matevosovp/practicum-sem4-praktika2) | **2,76 млн событий**, **20,28 млн свойств**; PostgreSQL, Airflow, MLflow, FastAPI | Recall@10 **0,1496** против **0,0085** у popularity baseline |
| [Модерация комментариев](projects/text-moderation/README.md) | TF-IDF, линейные модели, подбор порога, анализ ошибок | F1 **0,7817** на test исходного эксперимента |
| [Контрафакт и CLIP](projects/clip-counterfeit/README.md) | Классификация товаров, OpenCLIP, поиск похожих изображений | CatBoost F1-macro **0,9058** на validation; отдельные CLIP-эксперименты |
| [OCR + RAG для договоров](projects/document-ai/README.md) | PaddleOCR, структурный chunking, SentenceTransformers, Qdrant, API и мониторинг | Полный локальный цикл от скана до семантического поиска |
| [ML lifecycle недвижимости](https://github.com/matevosovp/Airflow-project) | ETL → DVC → [MLflow](https://github.com/matevosovp/Mlflow-project) → [FastAPI и Grafana](https://github.com/matevosovp/ML-model_deployment_in_a_cloud_infrastructure) | Пайплайны, контракты данных и автоматические проверки |

Результаты получены в учебных, самостоятельных и соревновательных проектах. Схемы offline-оценки описаны в проектах.

## Дополнительные направления

- [PyTorch: нейросетевая регрессия](projects/pytorch-regression/README.md): MLP, BatchNorm, Dropout, AdamW, early stopping.
- [Временные ряды и эконометрика](projects/time-series/README.md): ARIMA/AutoARIMA, стационарность, коинтеграция, временные признаки.
- [Uplift и клиентское поведение](projects/uplift/README.md): сравнение методов, Uplift@30 **0,0619**.
- [Dota 2](https://github.com/matevosovp/Dota-2-winner-prediction): табличные и JSONL-данные, CatBoost/Optuna, ROC-AUC CV около **0,84**, командный проект.
- Kaggle Playground S5E5: **163 / 4316, топ-4%**, ансамбль CatBoost, XGBoost и LightGBM.

## Технологии

| Направление | Инструменты |
|---|---|
| ML и аналитика | Python, pandas, NumPy, scikit-learn, CatBoost, XGBoost, LightGBM, Optuna, statsmodels |
| NLP / CV | PyTorch, OpenCLIP, TF-IDF, PaddleOCR, SentenceTransformers, Keras |
| Данные и MLOps | SQL, PostgreSQL, Airflow, DVC, MLflow, S3, Qdrant |
| Сервисы | FastAPI, Docker, Prometheus, Grafana, Git |

## Посмотреть код

Компактные реализации: [классификация текстов](portfolio/text_moderation.py), [CLIP-поиск](portfolio/clip_search.py), [PyTorch](portfolio/star_regression.py), [ARIMA](portfolio/forecast.py).

```bash
git clone https://github.com/matevosovp/matevosovp.git
cd matevosovp
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt pytest
python -m pytest -q
```

Для нейросетевых примеров: `python -m pip install -r requirements-dl.txt`. Проверки используют синтетические данные; команды для реальных данных приведены в описаниях проектов.

[![Portfolio checks](https://github.com/matevosovp/matevosovp/actions/workflows/portfolio-checks.yml/badge.svg)](https://github.com/matevosovp/matevosovp/actions/workflows/portfolio-checks.yml)

<details>
<summary>English summary</summary>

Data Scientist / ML Engineer with an MSc in Data Science in Economics and a Chemical Engineering degree from the University of Manchester. My project work covers recommender systems, tabular ML, text moderation, document AI, PyTorch and CLIP, with reproducible evaluation and deployment using Airflow, MLflow, FastAPI and Docker. English: C2. See the projects for methods, evaluation protocols and runnable code.

</details>
