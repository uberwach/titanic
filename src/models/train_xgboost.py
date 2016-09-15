#!/usr/bin/env python
# -*- coding: utf-8 -*-
import click

import pandas as pd
from sklearn.externals import joblib
import xgboost as xgb

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('model_path', type=click.Path())
def main(input_filepath, model_path):
    data = joblib.load(input_filepath)
    df_train = data['train']
    # df_test = data['test']

    print("Training shape: %s" % str(df_train.shape))
    # joblib.dump(clf, model_path)

    X_train = df_train.drop(['Survived'], axis=1).values.astype(float)
    y_train = df_train['Survived'].values.astype(float)

    params = {
        'objective': 'binary:logitraw',
        'max_depth': 3,
        'n_estimators': 500,
        'eta': 0.05,
        'colsample_bytree': 0.8,
        'subsample': 0.72,
        'eval_metric': 'error',
        'silent': 1,
        'gamma': 1
    }

    Z_train = xgb.DMatrix(X_train, label=y_train)

    xgb.cv(params=params, dtrain=Z_train,
           num_boost_round=300,
           nfold=10,
           verbose_eval=10,
           seed=42)

    clf = xgb.train(params=params, dtrain=Z_train,
                    num_boost_round=1000,
                    verbose_eval=10)

    joblib.dump(clf, model_path)
if __name__ == '__main__':
    main()
