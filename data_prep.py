import os
import pandas as pd
import yaml
import dvc.api

## For output data
OUTPUT_PATH = os.path.join(os.getcwd(), 'outs')
os.makedirs(OUTPUT_PATH, exist_ok=True)

## Read the Dataset
TRAIN_PATH = os.path.join(os.getcwd(), 'dataset.csv')
df = pd.read_csv(TRAIN_PATH)


def prepare_fn(age_threshold: int):
    ## Drop first 3 features
    df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

    ## Filtering using Age Feature using threshold
    df.drop(index=df[df['Age'] > age_threshold].index.tolist(), axis=0, inplace=True)

    ## Dump the DF
    df.to_csv(os.path.join(OUTPUT_PATH, 'dataset_cleaned.csv'), index=False)
    
def main():
    with open("params.yaml")as f:
        prepare_params = yaml.safe_load(f)['prepare']
    # or     
    # prepare_params = dvc.api.params_show()['prepare']
    
    
    AGE_THRESHOLD: int = prepare_params['age_threshold']   
    prepare_fn(age_threshold=AGE_THRESHOLD)
    
    
if __name__ == '__main__':
    main()