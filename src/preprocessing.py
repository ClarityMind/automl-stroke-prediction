import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    imputer = SimpleImputer(strategy='median')
    df['bmi'] = imputer.fit_transform(df[['bmi']])
    return df

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=['object']).columns
    binary_cols = [col for col in cat_cols if df[col].nunique() == 2]
    multi_cat_cols = [col for col in cat_cols if df[col].nunique() > 2]

    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)
    return df

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

def apply_pca(df: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(df)
    return pd.DataFrame(df_pca)

def preprocess_pipeline(df: pd.DataFrame, use_pca: bool = False, pca_components: int = 5) -> pd.DataFrame:
    df = df.drop(columns=['id'])
    df = handle_missing_values(df)
    df = encode_categorical(df)
    df = scale_features(df)
    if use_pca:
        df = apply_pca(df, pca_components)
    return df