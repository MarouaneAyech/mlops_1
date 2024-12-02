# %% [markdown]
# Import bibliothèques

# %%
## main lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
import os

## skelarn -- preparation & processing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

## sklearn -- modeling
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression


## sklearn -- metrics
from sklearn.metrics import mean_squared_error, r2_score

# %% [markdown]
# Data Processing

# %%
# Définition de la fonction personnalisée
def replace_outliers(data, low, high, replacement):
    '''
    data : is a Series (Dataframe column)
    '''
    # Effectuer l'imputation des outliers sur la colonne
    colonne_transformee = data.apply(lambda x: replacement if x < low or x > high else x)
    # retourner un 2D array
    return colonne_transformee.values.reshape(-1,1)

# Définir outlier imputer
outlier_imputer_num = ColumnTransformer(
    transformers=[  ('size_m2', 
                    FunctionTransformer(replace_outliers, 
                                        kw_args={'low': 10, 'high': 500, 'replacement': 140}), 
                    'size_m2'),
        
                    ('num_bedrooms',
                    FunctionTransformer(replace_outliers, 
                                        kw_args={'low': 0, 'high': 10, 'replacement': 3}), 
                    'num_bedrooms'),

                    ('num_bathrooms',
                    FunctionTransformer(replace_outliers, 
                                        kw_args={'low': 0, 'high': 10, 'replacement': 2}),
                    'num_bathrooms')
    ],
    remainder='passthrough'
)

# %%
cols_num = ['size_m2', 'num_bedrooms', 'num_bathrooms', 'distance_school','public_transport_access', 'property_tax']

def to_dataframe_with_columns(X, columns):
    # Retourner un dataframe
    return pd.DataFrame(X, columns=columns)

pipeline_num = Pipeline(steps=[
    ('nan_imputer_num', SimpleImputer(strategy='mean')),  # Imputation des NaN
    # ('to_dataframe', FunctionTransformer(to_dataframe_with_columns, kw_args={'columns': cols_num})),
    # ('outlier_imputer_num', outlier_imputer_num)
])

# %%
def extract_year(date): 
    return pd.to_datetime(date).dt.year.values.reshape(-1, 1)

# Définir le ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('pipeline_num', pipeline_num, ['size_m2', 'num_bedrooms', 'num_bathrooms', 'distance_school','public_transport_access', 'property_tax']),
        ('onehot', OneHotEncoder(sparse_output=False), ['city']),  # OneHotEncoder sur la colonne catégorielle
        ('year', FunctionTransformer(extract_year), 'date_built')  # Extraction de l'année sur la colonne date
    ]
)

# %% [markdown]
# Pipeline global

# %%
pipeline_global = Pipeline(steps=[
    ('preprocessor', preprocessor),          # Étape de prétraitement
    ('scaling', StandardScaler()),            # Normalisation
    ('pca', PCA()),              # Réduction dimensionnelle
    ('regressor', LinearRegression())
])
pipeline_global

# %% [markdown]
# Load dataset CSV

# %%
df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'data.csv'))

# %% [markdown]
# Split dataset into features (X) & target (y)

# %%
X = df.drop('price',axis=1, errors='ignore')
y = df['price'].copy()

# %% [markdown]
# Split dataset into train & test 

# %%
X_train, X_test, y_train, y_test = train_test_split(X,y, 
                                                    train_size=0.7, 
                                                    random_state=45)

# %% [markdown]
# Tuning des hyperparameters avec GridSaerch

# %%
param_grid = {
    'pca__n_components': [2, 3 , 4, 5, 6],  # Essayer différentes valeurs pour PCA
}

# %%
grid_search = GridSearchCV(estimator=pipeline_global, param_grid=param_grid, cv=3, scoring='r2')

# %%
grid_search.fit(X_train, y_train)

# %% [markdown]
# Evaluate the pipeline

# %%
# Rapport détaillé du tuning (toutes les pipelines triés par ordre decroissant)
cv_results = pd.DataFrame(grid_search.cv_results_).sort_values(by='rank_test_score')
print(cv_results[['params', 'mean_test_score', 'std_test_score','rank_test_score']])

# %%
# Meilleurs paramètres et scores de Gridsearch
print("Meilleurs paramètres :", grid_search.best_params_)
print("Meilleur score de validation R² :", grid_search.best_score_)
print("Meilleur score de validation R² :", grid_search.best_score_)

# Évaluer sur les données de train
train_score = grid_search.score(X_train, y_train)
print("Score R² sur train :", train_score)

# Évaluer sur les données de test
test_score = grid_search.score(X_test, y_test)
print("Score R² sur test :", test_score)

# %%
# Meilleur pipeline
best_pipeline = grid_search.best_estimator_

# Évaluer sur les données de train
train_score = best_pipeline.score(X_train, y_train)
print("Score R² sur train :", train_score)

# Évaluer sur les données de test
test_score = best_pipeline.score(X_test, y_test)
print("Score R² sur test :", test_score)

# %%
# save metrics
with open('metrics.txt','w') as f :
    f.write(f"Score R2 sur train : {train_score:.2f}\n")
    f.write(f"Score R2 sur test : {test_score:.2f}\n")

# %%
# save model
dump(best_pipeline, os.path.join(os.getcwd(), 'models', "best_pipeline.pkl"))


