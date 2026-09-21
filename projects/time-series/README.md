# Временные ряды и эконометрика

Учебная практика: ARIMA/AutoARIMA, ACF/PACF, диагностика остатков, ADF/KPSS/Phillips-Perron/DF-GLS, коинтеграция; временные признаки и Random Forest / Gradient Boosting.

[Пример ARIMA](../../portfolio/forecast.py) сравнивает ARIMA(1,1,1) с last-value и seasonal-naive на последнем временном блоке. Порядок модели фиксирован до оценки.

```bash
python -m pip install -r requirements.txt
python -m portfolio.forecast --data data/series.csv --column value --horizon 12 --season-length 12
```

CSV предварительно сортируется по времени и содержит регулярный числовой ряд без пропусков. Вывод: MAE/RMSE трёх подходов. Горизонт и сезонность задаются в наблюдениях.

Дополнительные работы: регрессии statsmodels, VIF, White/Breusch-Pagan/Goldfeld-Quandt; учебное моделирование прибыли при закупке скоропортящегося товара со случайным спросом.

Это учебный опыт, не промышленное управление запасами или CLTV. Для реального прогноза нужны rolling-origin backtesting и контроль момента доступности внешних факторов.
