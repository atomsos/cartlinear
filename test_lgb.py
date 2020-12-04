import pandas as pd
import lightgbm as lgb
import wandb
from wandb.lightgbm import wandb_callback

wandb.init(
    magic=True,
    project='cartlinear',
    name='lightgbm-gbdt'
)
# wandb.log()


dataset = pd.DataFrame([{'x': _, 'y': _} for _ in range(100000)]).astype('float')

ndata = len(dataset)
ntrain = int(ndata * 0.8)
nvalid = ndata - ntrain

def run_model(dataset):
    train_idx, valid_idx = list(range(ntrain)), list(range(ntrain, ndata))
    train, valid = dataset.iloc[train_idx], dataset.iloc[valid_idx]
    train_x, train_y = train[['x']], train['y']
    valid_x, valid_y = valid[['x']], valid['y']
    n_train = lgb.Dataset(train_x, label=train_y)
    n_valid = lgb.Dataset(valid_x, label=valid_y)
    params = {
        'learning_rate': 0.2,
        'boosting_type': 'gbdt', 
        #  ‘gbdt’, traditional Gradient Boosting Decision Tree. 
        # ‘dart’, Dropouts meet Multiple Additive Regression Trees. 
        # ‘goss’, Gradient-based One-Side Sampling. 
        # ‘rf’, Random Forest.
        'objective': 'regression',
        'max_depth': 7,
        'num_leaves': 100,
        'metric': 'mse',
        'feature_fraction': 0.8,
        'bagging_fraction': 0.6,
        'bagging_freq': 6,
        'bagging_seed': 1,
        'seed': 8,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 50,
        'verbose': 1,
#         'device_type': 'gpu',
        'nthread': 6,
        'lambda_l2': 0.01,
    }

    clf = lgb.train(
        params=params,
        train_set=n_train,
        num_boost_round=35000,
        valid_sets=[n_valid, n_train],
        early_stopping_rounds=500,
        verbose_eval=200,
        callbacks=[wandb_callback()],
    )
    # train_pred[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)



if __name__ == '__main__':
    run_model(dataset)
