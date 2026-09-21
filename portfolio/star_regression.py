"""PyTorch MLP regression with a separate validation and test split."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib


class StarMLP(nn.Module):
    def __init__(self, input_size, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_size, 32), nn.BatchNorm1d(32), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(32, 16), nn.BatchNorm1d(16),
                                     nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.network(x).squeeze(-1)


def train(frame, target, output, epochs=500, patience=30):
    if target not in frame or frame[target].isna().any():
        raise ValueError("Target must exist and contain no missing values")
    if len(frame) < 30 or epochs < 1 or patience < 1:
        raise ValueError("Require at least 30 rows and positive epochs/patience")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(1)
    features = frame.drop(columns=target)
    y = frame[target].to_numpy(dtype=np.float32)
    train_idx, test_idx = train_test_split(np.arange(len(frame)), test_size=0.2, random_state=42)
    train_idx, valid_idx = train_test_split(train_idx, test_size=0.25, random_state=42)
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in features if c not in numeric]
    preprocessing = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("categorical", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore", sparse_output=False)), categorical),
    ])
    x_train = preprocessing.fit_transform(features.iloc[train_idx]).astype(np.float32)
    x_valid = torch.tensor(preprocessing.transform(features.iloc[valid_idx]).astype(np.float32))
    x_test = torch.tensor(preprocessing.transform(features.iloc[test_idx]).astype(np.float32))
    mean, scale = float(y[train_idx].mean()), max(float(y[train_idx].std()), 1e-6)
    y_train = torch.tensor((y[train_idx] - mean) / scale)
    y_valid = torch.tensor((y[valid_idx] - mean) / scale)
    batch_size = min(32, len(train_idx))
    loader = DataLoader(TensorDataset(torch.tensor(x_train), y_train), batch_size=batch_size,
                        shuffle=True, drop_last=(len(train_idx) % batch_size == 1))
    model = StarMLP(x_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    best, best_state, waiting, best_epoch = float("inf"), None, 0, 0
    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(x_valid), y_valid))
        if validation_loss < best:
            best, best_state, waiting, best_epoch = validation_loss, copy.deepcopy(model.state_dict()), 0, epoch + 1
        else:
            waiting += 1
        if waiting >= patience:
            break
    if best_state is None:
        raise ValueError("Training failed to produce finite validation loss")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = model(x_test).numpy() * scale + mean
    report = {"seed": 42, "best_epoch": best_epoch, "test_rmse": float(np.sqrt(np.mean((predictions-y[test_idx])**2))),
              "rows": {"train":len(train_idx),"validation":len(valid_idx),"test":len(test_idx)},
              "evaluation":"60/20/20 random split; preprocessing and target scaling fit on train only"}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict":best_state,"input_size":x_train.shape[1],"target_mean":mean,"target_scale":scale}, output/'model.pt')
    joblib.dump(preprocessing, output/'preprocessing.joblib')
    (output/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--target',default='Temperature (K)')
    parser.add_argument('--output',type=Path,default='artifacts/stars')
    parser.add_argument('--epochs',type=int,default=500)
    args=parser.parse_args()
    print(json.dumps(train(pd.read_csv(args.data),args.target,args.output,args.epochs),indent=2))
