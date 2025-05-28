import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

dataset = pd.read_csv('data/raw/healthcare-dataset-stroke-data.csv')

dataset.drop(columns=['id'], inplace=True)

imputer = SimpleImputer(strategy='median')
dataset['bmi'] = imputer.fit_transform(dataset[['bmi']])

cat_cols = dataset.select_dtypes(include=['object']).columns

binary_cols = [col for col in cat_cols if dataset[col].nunique() == 2]
multi_cat_cols = [col for col in cat_cols if dataset[col].nunique() > 2]

le = LabelEncoder()
for col in binary_cols:
    dataset[col] = le.fit_transform(dataset[col])

dataset = pd.get_dummies(dataset, columns=multi_cat_cols, drop_first=True)

num_cols = dataset.select_dtypes(include=['float64', 'int64']).columns

scaler = StandardScaler()
dataset[num_cols] = scaler.fit_transform(dataset[num_cols])

dataset.to_csv('data/processed/healthcare-dataset-stroke-data-processed.csv', index=False)