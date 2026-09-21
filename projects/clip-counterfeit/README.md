# Контрафакт и поиск похожих товаров с CLIP

Проект на данных соревнования Ozon 2025: табличная классификация контрафакта и отдельные эксперименты с визуальными и текстовыми представлениями.

| Эксперимент | Реализация | Результат |
|---|---|---|
| CatBoost | Табличная классификация | Accuracy **0,9773**, F1-macro **0,9058**, validation 39 440 объектов |
| CLIP / OpenCLIP | Энкодер брендов и поиск похожих изображений | Код энкодера и выполненная визуализация top-10 соседей |

Метрика CatBoost не относится к CLIP. Дообучение CLIP и прирост классификации от его эмбеддингов не заявляются.

[Код](../../portfolio/clip_search.py): ViT-B-32-quickgelu, локальные веса, пакетный PyTorch inference, L2-нормализация, cosine top-k без полной квадратной матрицы, исключение запроса из соседей.

```bash
python -m pip install -r requirements.txt
python -m portfolio.clip_search search --embeddings data/image_embeddings.npy --query-index 0 --top-k 10
```

Матрица `.npy`: `(число товаров, размерность)`, строки соответствуют вашему индексу товаров. Для текстового энкодера:

```bash
python -m pip install -r requirements-dl.txt
python -m portfolio.clip_search encode-brands --brands data/brands.txt --checkpoint data/open_clip_model.safetensors --output artifacts/brands.npy
```

Один бренд на строку. Checkpoint должен соответствовать архитектуре; текстовые и визуальные векторы сопоставляются только при согласованных весах. Данные соревнования и веса не публикуются.

Тест сравнивает top-k с полным сортированием cosine scores. OpenCLIP с реальными весами требует отдельного запуска; CI не скачивает checkpoint. Следующий исследовательский шаг: late fusion табличных и CLIP-признаков с независимой оценкой.
