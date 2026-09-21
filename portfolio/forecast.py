"""Chronological ARIMA evaluation against last-value and seasonal baselines."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def evaluate(values, horizon=12, order=(1,1,1), season_length=12):
    values=np.asarray(values,dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Require a finite one-dimensional series')
    if horizon < 1 or season_length < 1 or len(values)-horizon < max(20,season_length):
        raise ValueError('Insufficient history or invalid horizon/season length')
    train,test=values[:-horizon],values[-horizon:]
    forecast=ARIMA(train,order=order).fit().forecast(horizon)
    candidates={'arima':forecast,'last_value':np.repeat(train[-1],horizon),
                'seasonal_naive':np.resize(train[-season_length:],horizon)}
    results={}
    for name,pred in candidates.items():
        results[name]={'mae':float(np.abs(test-pred).mean()),'rmse':float(np.sqrt(np.mean((test-pred)**2)))}
    return {'evaluation':'last chronological block; order fixed before test evaluation','train_size':len(train),'horizon':horizon,'order':list(order),'metrics':results}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--column',required=True)
    parser.add_argument('--horizon',type=int,default=12)
    parser.add_argument('--season-length',type=int,default=12)
    args=parser.parse_args()
    print(json.dumps(evaluate(pd.read_csv(args.data)[args.column],args.horizon,season_length=args.season_length),indent=2))
