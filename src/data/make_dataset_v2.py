# -*- coding: utf-8 -*-
import os
import click
import logging
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import numpy as np
import re

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
    nan_age_idx = pd.isnull(df['Age'])
    df['Age_nan'] = nan_age_idx.astype(int)
    # replace missing ages with mean
    mean_age = df['Age'].mean()
    df['Age'][nan_age_idx] = mean_age

    # Encode embarked as numbers 0, 1, 2, ...
    df = pd.get_dummies(df, columns=['Embarked'], dummy_na=True,
                        drop_first=True)

    df['Title'] = df['Name'].apply(extract_title)
    df = pd.get_dummies(df, columns=['Title'], dummy_na=True, drop_first=True)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
