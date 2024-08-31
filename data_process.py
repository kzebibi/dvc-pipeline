import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn_features.transformers import DataFrameSelector
import os



## For output data
OUTPUT_PATH = os.path.join(os.getcwd(), 'outs')
os.makedirs(OUTPUT_PATH, exist_ok=True)

df = pd.read_csv(os.path.join(OUTPUT_PATH, 'dataset_cleaned.csv'))

## To features and target
X = df.drop(columns=['Exited'], axis=1)
y = df['Exited']

## Split to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=45, stratify=y)


## --------------------- Data Processing ---------------------------- ##

## Slice the lists
num_cols = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary']
categ_cols = ['Gender', 'Geography']

ready_cols = list(set(X_train.columns.tolist()) - set(num_cols) - set(categ_cols))


## For Numerical
num_pipeline = Pipeline(steps=[
                        ('selector', DataFrameSelector(num_cols)),
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                    ])


## For Categorical
categ_pipeline = Pipeline(steps=[
                        ('selector', DataFrameSelector(categ_cols)),
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('ohe', OneHotEncoder(drop='first', sparse_output=False))
                    ])


## For ready cols
ready_pipeline = Pipeline(steps=[
                        ('selector', DataFrameSelector(ready_cols)),
                        ('imputer', SimpleImputer(strategy='most_frequent'))
                    ])



## combine all
all_pipeline = FeatureUnion(transformer_list=[
                                    ('numerical', num_pipeline),
                                    ('categorical', categ_pipeline),
                                    ('ready', ready_pipeline)
                                ])

## apply
all_pipeline.fit_transform(X_train)

out_categ_cols = categ_pipeline.named_steps['ohe'].get_feature_names_out(categ_cols)

X_train_final = pd.DataFrame(all_pipeline.fit_transform(X_train), columns=num_cols + list(out_categ_cols) + ready_cols)
X_test_final = pd.DataFrame(all_pipeline.transform(X_test), columns=num_cols + list(out_categ_cols) + ready_cols)

## Dump the data
X_train_final.to_csv(os.path.join(OUTPUT_PATH, 'processed_X_train.csv'), index=False)
X_test_final.to_csv(os.path.join(OUTPUT_PATH, 'processed_X_test.csv'), index=False)
y_train.to_csv(os.path.join(OUTPUT_PATH, 'processed_y_train.csv'), index=False)
y_test.to_csv(os.path.join(OUTPUT_PATH, 'processed_y_test.csv'), index=False)

