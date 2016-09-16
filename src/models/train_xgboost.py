#!/usr/bin/env python
# -*- coding: utf-8 -*-
import click

import pandas as pd
import numpy as np
from sklearn.externals import joblib
import xgboost as xgb

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('model_path', type=click.Path())
@click.argument('submission_path', type=click.Path())
def main(input_filepath, model_path, submission_path):
    data = joblib.load(input_filepath)
    df_train = data['train']

    print("Training shape: %s" % str(df_train.shape))
    # joblib.dump(clf, model_path)

    X_train = df_train.drop(['Survived'], axis=1).values.astype(float)
    y_train = df_train['Survived'].values.astype(float)

    # random shuffle data
    np.random.seed(42) # fix seed
    permute_idx = np.random.permutation(len(X_train))
    X_train, y_train = X_train[permute_idx], y_train[permute_idx]
    params = {
        'objective': 'binary:logistic',
        'max_depth': 2,
        'n_estimators': 40,
        'eta': 0.005,
        'colsample_bytree': 0.8,
        'subsample': 1.0,
        'eval_metric': 'error',
        'silent': 1,
        'min_child_weight': 1,
        'gamma': 1
    }

    Z_train = xgb.DMatrix(X_train, label=y_train)

    xgb.cv(params=params, dtrain=Z_train,
           num_boost_round=1000,
           nfold=10,
           verbose_eval=10,
           seed=42)


    clf = xgb.train(params=params, dtrain=Z_train,
                    num_boost_round=1000)

    # save model
    joblib.dump(clf, model_path)

    # create submission
    df_test = data['test']
    X_test = df_test.drop(['PassengerId', 'Survived'], axis=1).values.astype(float)

    y_pred = clf.predict(xgb.DMatrix(X_test))

    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0

    pd.DataFrame({
        'PassengerId': df_test['PassengerId'],
        'Survived': y_pred.astype(int)
    }).to_csv(submission_path, index=False)

if __name__ == '__main__':
    main()
