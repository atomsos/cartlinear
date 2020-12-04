import numpy as np
import pandas as pd
import wandb
import xgboost as xgb
from wandb.xgboost import wandb_callback

wandb.init(
    magic=True,
    project='cartlinear',
    name='xgboost-gbtree',
)
# wandb.log()

Xs= np.linspace(-10, 10, 100).reshape((-1, 1))
Ys= np.linspace(-20, 20, 100) + np.random.normal(loc=0, scale=3.5, size=(100, ))


dataset = pd.DataFrame([{'x': x, 'y': y} for (x, y) in zip(Xs, Ys)]).astype(np.float32)

ndata = len(dataset)
ntrain = int(ndata * 0.8)
nvalid = ndata - ntrain

def run_model(dataset):
    train_idx, valid_idx = list(range(ntrain)), list(range(ntrain, ndata))
    train, valid = dataset.iloc[train_idx], dataset.iloc[valid_idx]
    train_x, train_y = train[['x']], train['y']
    valid_x, valid_y = valid[['x']], valid['y']
    n_train = xgb.DMatrix(train_x, label=train_y)
    n_valid = xgb.DMatrix(valid_x, label=valid_y)
    params = {
        'objective': 'reg:squarederror',
        'eta': 0.1,
        'max_depth': 6,
        'nthread': 6,
        'booster': 'gbtree', # ``gbtree``, ``gblinear`` or ``dart``
    }

    clf = xgb.train(
        params=params,
        dtrain=n_train,
        num_boost_round=35000,
        # obj='reg:squarederror',
        # feval='reg:squarederror',
        evals=[(n_valid, 'valid'), (n_train, 'train')],
        early_stopping_rounds=500,
        verbose_eval=200,
        callbacks=[wandb_callback()],
    )



if __name__ == '__main__':
    run_model(dataset)
