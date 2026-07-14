# Павел Матевосов — ML Engineer / Data Scientist

Разрабатываю ML-решения полного цикла: от анализа данных и построения честного offline-эксперимента до воспроизводимого обучения, API, контейнеризации и мониторинга. Основные интересы — рекомендательные системы, табличный ML и MLOps.

Магистратура **«Data Science в экономике»**, РАНХиГС × Яндекс, 2024–2026.

## Ключевые проекты

| Проект | Что реализовано | Результат и технологии |
|---|---|---|
| [Рекомендации банковских продуктов](https://github.com/matevosovp/practicum-sem4-praktika1) | Multilabel ranking новых продуктов, time-based split, поэтапное сравнение baseline и CatBoost, MLflow Registry, FastAPI и Prometheus | **Test MAP@3: 0.6527** · Python, CatBoost, MLflow, FastAPI, Docker |
| [E-commerce recommender](https://github.com/matevosovp/practicum-sem4-praktika2) | Memory-safe обработка 20+ млн строк свойств, candidate generation, offline evaluation, ETL в PostgreSQL и контур переобучения | **Recall@10: 0.1496** против 0.0085 у popularity baseline · Python, Airflow, MLflow, PostgreSQL |
| [Предсказание победы в Dota 2](https://github.com/matevosovp/Dota-2-winner-prediction) | EDA, признаки из табличных данных и JSONL-логов, кросс-валидация без утечек, Optuna и анализ ошибок | **ROC-AUC CV: 0.84** · CatBoost, Optuna, pandas |
| [ML lifecycle для оценки недвижимости](https://github.com/matevosovp/Airflow-project) | Связанный набор проектов: ETL и очистка данных, DVC-пайплайн, управление экспериментами, Model Registry, сервис предсказаний и мониторинг | Airflow, DVC, MLflow, FastAPI, Docker Compose, Prometheus, Grafana |

## ML lifecycle: от данных до сервиса

Проекты по оценке недвижимости показывают последовательные этапы одной ML-системы:

1. [Airflow ETL + DVC](https://github.com/matevosovp/Airflow-project) — сбор и очистка данных из PostgreSQL, обучение модели и версионирование артефактов в S3.
2. [MLflow Tracking + Model Registry](https://github.com/matevosovp/Mlflow-project) — feature engineering, отбор признаков, подбор гиперпараметров и регистрация версий модели.
3. [Production-сервис](https://github.com/matevosovp/ML-model_deployment_in_a_cloud_infrastructure) — REST API на FastAPI, контейнеризация и наблюдаемость через Prometheus и Grafana.

## Технический стек

- **ML и анализ данных:** Python, pandas, NumPy, scikit-learn, CatBoost, Optuna
- **Рекомендательные системы:** ranking, candidate generation, time-based evaluation, MAP@K, Recall@K, NDCG@K
- **MLOps и Data Engineering:** MLflow, Airflow, DVC, PostgreSQL, S3
- **Backend и инфраструктура:** FastAPI, Docker, Docker Compose, Prometheus, Grafana
- **Инструменты:** SQL, Git, Jupyter

## Как я подхожу к ML-задачам

- начинаю с понятного baseline и связываю offline-метрики с бизнес-сценарием;
- разделяю данные по времени там, где случайный split создаёт утечку;
- фиксирую параметры, артефакты и результаты экспериментов;
- думаю не только о качестве модели, но и о воспроизводимости, serving и мониторинге.

## Контакты

- Email: [matevosovp@gmail.com](mailto:matevosovp@gmail.com)
- GitHub: [@matevosovp](https://github.com/matevosovp)
