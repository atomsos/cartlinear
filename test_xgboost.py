import pandas as pd
import xgboost as xgb
import wandb
from wandb.xgboost import wandb_callback

wandb.init(
    magic=True,
    project='cartlinear',
    name='xgboost-gbtree'
)


dataset = pd.DataFrame([{'x': _, 'y': _} for _ in range(100000)]).astype('float')

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
    # params = {
    #     'learning_rate': 0.002,
    #     'boosting_type': 'gbdt',
    #     'objective': 'regression',
    #     'num_depth': 7,
    #     'num_leaves': 100,
    #     'metric': 'mse',
    #     'feature_fraction': 0.8,
    #     'bagging_fraction': 0.6,
    #     'bagging_freq': 6,
    #     'seed': 8,
    #     'bagging_seed': 1,
    #     'feature_fraction_seed': 7,
    #     'min_data_in_leaf': 50,
    #     'verbose': 1,
#   #       'device_type': 'gpu',
    #     'nthread': 6,
    #     'lambda_l2': 0.01,
    # }

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
