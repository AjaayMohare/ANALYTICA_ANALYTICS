import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv("datasets.csv")
df.replace('?', np.nan, inplace=True)
df.columns = df.columns.str.strip()
df.rename(columns={'pcv': 'packed_cell_volume', 
                   'wbcc': 'white_blood_cell_count', 
                   'rbcc': 'red_blood_cell_count'}, inplace=True)

numeric_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 
                'hemo', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df_processed = df.copy()

target_le = LabelEncoder()
df_processed['class'] = target_le.fit_transform(df_processed['class'])

categorical_cols = df_processed.select_dtypes(include='object').columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col] = df_processed[col].astype(str)
    df_processed[col] = le.fit_transform(df_processed[col])

df_processed['bu_sc_ratio'] = df_processed['bu'] / df_processed['sc']
df_processed['hemo_pcv_ratio'] = df_processed['hemo'] / df_processed['packed_cell_volume']
df_processed['anemia_indicator'] = (df_processed['hemo'] < 13).astype(int)
df_processed['abnormality_count'] = df_processed[['pcc', 'ba', 'htn', 'dm', 'cad', 'pe', 'ane']].sum(axis=1)

# 3. Data Splitting
X = df_processed.drop('class', axis=1)
y = df_processed['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

imputer = KNNImputer(n_neighbors=5)
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

final_model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(C=1, solver='liblinear', random_state=42, max_iter=1000))
])

final_model_pipeline.fit(X_train, y_train)

model_filename = 'SUB_CHALLENGE_MODEL.pkl'
joblib.dump(final_model_pipeline, model_filename)

print(f"Model training complete. Final model saved as '{model_filename}'.")