#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import click
import logging
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import numpy as np
import re
from sklearn.externals import joblib
# This program reads in both train and test data set
# and creates a dataset dictionary
# of cleaned and sanitized data.
# result format:
# {
#  'train': <pandas.DataFrame>
#  'test': <pandas.DataFrame>
# }


# extracts title from a name, i.e.
# extract_title('Caldwell, Mr. Albert Francis') = 'Mr.'
def extract_title(name):
    m = re.search('[^,]+, ([^\.]+)\..*', name)
    return m.group(1)


@click.command()
@click.argument('train_filepath', type=click.Path(exists=True))
@click.argument('test_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(train_filepath, test_filepath, output_filepath):
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    df_train = pd.read_csv(train_filepath, dtype={'Age': np.float64})
    df_test = pd.read_csv(test_filepath, dtype={'Age': np.float64})

    # combine into one data frame to process at once
    df = pd.concat([df_train, df_test])
    # keep information which age entries were NaN (helpful for some learners
    # think logistic regression vs decision trees)
    df['Age_nan'] = pd.isnull(df['Age']).astype(int)

    # we set Fare to the median for missing entries
    df.Fare.fillna(8.05, inplace=True)
    # Encode 'Embarked' as one-hot
    df.Embarked.fillna(value='S', inplace=True)

    df = pd.get_dummies(df, columns=['Embarked'],
                        drop_first=True)

    # same for 'Sex'
    df = pd.get_dummies(df, columns=['Sex'], dummy_na=True, drop_first=True)

    # Pclass
    df = pd.get_dummies(df, columns=['Pclass'], dummy_na=True, drop_first=True)
    # SibSp
    df = pd.get_dummies(df, columns=['SibSp'], dummy_na=True, drop_first=True)

    # extract title from name
    df['Title'] = df['Name'].apply(extract_title)
    df.loc[df['Title'] == 'Ms', 'Title'] = 'Miss'

    for t in df[np.isnan(df["Age"])].Title.unique():
        df.loc[(df["Title"] == t) & np.isnan(df["Age"]), "Age"] = df[
            df["Title"] == t].Age.median()

    df = pd.get_dummies(df, columns=['Title'], dummy_na=True, drop_first=True)

    # clean up unused columns
    df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    # split data frame again to store seperately
    df_train = df[~pd.isnull(df['Survived'])]
    df_test = df[pd.isnull(df['Survived'])]

    df_train.drop(['PassengerId'], axis=1, inplace=True)
    logger.info('Column names: {}'.format(df_train.columns))

    joblib.dump({
        'train': df_train,
        'test': df_test
    }, output_filepath, compress=3)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
