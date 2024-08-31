import os
import pandas as pd


## For output data
OUTPUT_PATH = os.path.join(os.getcwd(), 'outs')
os.makedirs(OUTPUT_PATH, exist_ok=True)

## Read the Dataset
TRAIN_PATH = os.path.join(os.getcwd(), 'dataset.csv')
df = pd.read_csv(TRAIN_PATH)

## Drop first 3 features
df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

## Filtering using Age Feature using threshold
df.drop(index=df[df['Age'] > 80].index.tolist(), axis=0, inplace=True)

## Dump the DF
df.to_csv(os.path.join(OUTPUT_PATH, 'dataset_cleaned.csv'), index=False)