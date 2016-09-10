# -*- coding: utf-8 -*-
import os
import click
import logging
from dotenv import find_dotenv, load_dotenv

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib


def read_data(df):
    logging.info(df.columns)
    X = df.drop(["Survived", "PassengerId"], axis=1).values
    y = df["Survived"]

    return X, y

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('model_path', type=click.Path())
def main(input_filepath, model_path):
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    df = pd.read_csv(input_filepath)

    X, y = read_data(df)
    logging.info("Input dimension: {}".format(X.shape))
    logging.info("Training Logistic Regression")
    clf = LogisticRegression(C=1e1)
    clf.fit(X, y)

    joblib.dump(clf, model_path)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
