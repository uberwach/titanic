.PHONY: clean lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')

#################################################################################
# COMMANDS                                                                      #
#################################################################################

requirements:
	pip install -q -r requirements.txt

data/interim/data_v2.pkl: data/raw/train.csv data/raw/test.csv requirements
	python src/data/make_dataset_v2.py data/raw/train.csv data/raw/test.csv \
		data/interim/data_v2.pkl

data/interim/processed_train.csv: data/raw/train.csv requirements
	python src/data/make_dataset.py data/raw/train.csv data/interim/processed_train.csv

data/interim/processed_test.csv: data/raw/test.csv requirements
	python src/data/make_dataset.py data/raw/test.csv data/interim/processed_test.csv

models/logreg_simple.pkl: data/interim/processed_train.csv requirements
	python src/models/train_model.py data/interim/processed_train.csv \
		models/logreg_simple.pkl

models/xgboost.pkl: data/interim/data_v2.pkl requirements
	python src/models/train_xgboost.py data/interim/data_v2.pkl models/xgboost.pkl

submissions/submission1.csv: data/interim/processed_test.csv models/logreg_simple.pkl requirements
	python src/models/predict_model.py data/interim/processed_test.csv \
		models/logreg_simple.pkl \
    submissions/submission1.csv

clean:
	find . -name "*.pyc" -exec rm {} \;

lint:
	flake8 --exclude=lib/,bin/,docs/conf.py .

sync_data_to_s3:
	aws s3 sync data/ s3://$(BUCKET)/data/

sync_data_from_s3:
	aws s3 sync s3://$(BUCKET)/data/ data/



#################################################################################
# PROJECT RULES                                                                 #
################################################################################
