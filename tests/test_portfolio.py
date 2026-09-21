import json
import numpy as np
import pandas as pd
import pytest
from portfolio.clip_search import nearest_neighbors
from portfolio.text_moderation import clean_text, train as train_text
from portfolio.forecast import evaluate


def test_retrieval_matches_brute_force_and_excludes_query():
    vectors=np.random.default_rng(42).normal(size=(100,16))
    actual=nearest_neighbors(vectors,4,10)
    normalized=vectors/np.linalg.norm(vectors,axis=1,keepdims=True)
    scores=normalized@normalized[4]
    scores[4]=-np.inf
    assert [r['row'] for r in actual]==np.argsort(-scores)[:10].tolist()
    assert 4 not in [r['row'] for r in actual]


def test_zero_embedding_is_rejected():
    with pytest.raises(ValueError,match='Zero'):
        nearest_neighbors([[0,0],[1,0]],0,1)


def test_text_training_artifacts(tmp_path):
    rows=[{'text':f'{"kind helpful" if i%2==0 else "rude hostile"} example {i}','toxic':i%2} for i in range(100)]
    report=train_text(pd.DataFrame(rows),tmp_path)
    assert sum(report['rows'].values())==100
    assert 0<=report['test_f1']<=1
    assert (tmp_path/'model.joblib').is_file()
    assert json.loads((tmp_path/'metrics.json').read_text())==report
    assert clean_text('<b>Hello</b> https://example.com')=='hello'


def test_forecast_chronological_contract():
    report=evaluate(np.arange(60,dtype=float)+np.sin(np.arange(60)),horizon=8,season_length=4)
    assert report['train_size']==52
    assert set(report['metrics'])=={'arima','last_value','seasonal_naive'}
    assert all(np.isfinite(m['rmse']) for m in report['metrics'].values())


def test_torch_training_smoke(tmp_path):
    pytest.importorskip('torch')
    from portfolio.star_regression import train
    rng=np.random.default_rng(4)
    frame=pd.DataFrame({'radius':rng.uniform(1,3,60),'color':['blue','red']*30})
    frame['temperature']=3000+500*frame.radius
    result=train(frame,'temperature',tmp_path,epochs=3)
    assert result['rows']=={'train':36,'validation':12,'test':12}
    assert np.isfinite(result['test_rmse'])
    assert (tmp_path/'model.pt').is_file()
