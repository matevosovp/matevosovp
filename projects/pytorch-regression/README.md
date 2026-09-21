# Прогноз температуры звёзд на PyTorch

Учебная регрессия по физическим и категориальным характеристикам 240 звёзд.

Архитектура: вход → Linear(32) → BatchNorm → ReLU → Dropout → Linear(16) → BatchNorm → ReLU → Linear(1). DataLoader, AdamW, MSELoss, early stopping, лучший checkpoint. В исходном проекте исследовались batch size и dropout.

[Воспроизводимая версия](../../portfolio/star_regression.py) использует train/validation/test 60/20/20. Импутация, кодирование и масштабирование fit только на train. Лучшая эпоха выбирается на validation, затем вычисляется test RMSE.

```bash
python -m pip install -r requirements-dl.txt
python -m portfolio.star_regression --data data/stars.csv --target "Temperature (K)" --epochs 500
```

Одна строка CSV соответствует звезде. Удалите индексы и признаки, вычисленные из target. Target числовой, без пропусков; остальные признаки обрабатываются автоматически.

Артефакты: `model.pt`, `preprocessing.joblib`, `metrics.json` в `artifacts/stars`. CPU используется для простого запуска. CI выполняет короткое обучение на синтетических данных.

Историческая RMSE не приводится: в исходной работе оценочная выборка участвовала в подборе параметров. Новая версия исправляет это разделением validation/test и сохранением лучшего состояния модели. Для маленькой выборки нужны повторные splits или nested CV и простой baseline.

[Архив исходного исследования](research.ipynb): код и пояснения учебной работы; сохранённые выводы очищены перед публикацией.
