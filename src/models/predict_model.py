# -*- coding: utf-8 -*-
import os
import click
import logging
from dotenv import find_dotenv, load_dotenv

import pandas as pd
from sklearn.externals import joblib

def read_data(df):
    logging.info(df.columns)
    X = df.drop(["Survived", "PassengerId"], axis=1).values
    y = df["Survived"]

    return X, y

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('submission_path', type=click.Path())
def main(input_filepath, model_path, submission_path):
    logger = logging.getLogger(__name__)

    df = pd.read_csv(input_filepath)

    X = df.drop("PassengerId", axis=1).values
    logging.info("Input dimension: {}".format(X.shape))

    model = joblib.load(model_path)
    y_pred = model.predict(X)

    df_pred = pd.DataFrame({
        'PassengerId': df['PassengerId'],
        'Survived': y_pred
    })

    df_pred.to_csv(submission_path, index=False)
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
