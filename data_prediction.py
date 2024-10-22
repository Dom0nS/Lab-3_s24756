import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
import logging

# Konfiguracja loggera
logging.basicConfig(
    filename='data_prediction.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

# Wczytanie datasetu
df = pd.read_csv('./CollegeDistance.csv')

# Podział zmiennych
num_cols = ['unemp', 'wage', 'distance', 'tuition', 'education']
cat_cols = ['gender', 'ethnicity', 'fcollege', 'mcollege', 'home', 'urban', 'income', 'region']

# Zmienna przewidywana (target)
target = 'score'

# Tworzymy pipeline dla zmiennych numerycznych i kategorycznych
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Imputacja wartości brakujących
    ('scaler', StandardScaler())  # Standaryzacja
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Imputacja najczęściej występujących wartości
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encoding
])

# Połączenie transformacji dla zmiennych numerycznych i kategorycznych
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

# Przygotowanie danych do modelu
X = df.drop(columns=[target])  # Zbiór cech (bez zmiennej score)
y = df[target]  # Zmienna przewidywana

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transformacja danych treningowych i testowych
X_train_prepared = preprocessor.fit_transform(X_train)
X_test_prepared = preprocessor.transform(X_test)

# Przekształcone dane są w formacie numpy array, zamieniamy je na pandas DataFrame
transformed_feature_names = (num_cols + 
                             list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols)))

X_train_prepared_df = pd.DataFrame(X_train_prepared, columns=transformed_feature_names)
X_test_prepared_df = pd.DataFrame(X_test_prepared, columns=transformed_feature_names)

# Wypisanie kilku przykładów danych po transformacji
logging.info("Przykładowe rekordy po transformacji (zbiór treningowy):")
logging.info(f"\n{X_train_prepared_df.head()}")

logging.info("Przykładowe rekordy po transformacji (zbiór testowy):")
logging.info(f"\n{X_test_prepared_df.head()}")

# Informacje o przygotowaniu danych
logging.info(f"Rozmiar danych treningowych: {X_train.shape[0]} próbek")
logging.info(f"Rozmiar danych testowych: {X_test.shape[0]} próbek")
logging.info(f"Liczba cech po inżynierii cech: {X_train_prepared.shape[1]}")
logging.info("Dane zostały przekształcone przy użyciu standaryzacji dla zmiennych numerycznych oraz one-hot encoding dla zmiennych kategorycznych.")

#Trenowanie modelu Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Trenowanie modelu na zbiorze treningowym
rf_model.fit(X_train_prepared, y_train)

# Przewidywanie na zbiorze treningowym
y_train_pred = rf_model.predict(X_train_prepared)

# Przewidywanie na zbiorze testowym
y_test_pred = rf_model.predict(X_test_prepared)

# Ocena modelu - MSE, MAE, RMSE, R^2, MAPE
train_mse = mean_squared_error(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)
train_mape = mean_absolute_percentage_error(y_train, y_train_pred)

test_mse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)
test_mape = mean_absolute_percentage_error(y_test, y_test_pred)

# Logowanie wyników
logging.info("Model Random Forest został wytrenowany.")
logging.info(f"Wyniki dla zbioru treningowego: MSE = {train_mse:.4f}, MAE = {train_mae:.4f}, RMSE = {train_rmse:.4f}, R^2 = {train_r2:.4f}, MAPE = {train_mape:.4f}")
logging.info(f"Wyniki dla zbioru testowego: MSE = {test_mse:.4f}, MAE = {test_mae:.4f}, RMSE = {test_rmse:.4f}, R^2 = {test_r2:.4f}, MAPE = {test_mape:.4f}")

# Rozpoczęcie optymalizacji na nowym modelu podstawowym
rf_base = RandomForestRegressor(random_state=42)

# Parametry do tuningu
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Użycie GridSearchCV do tuningu hiperparametrów
grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# Dopasowanie modelu na danych treningowych
grid_search.fit(X_train_prepared, y_train)

# Najlepsze parametry
best_params = grid_search.best_params_

# Logowanie najlepszych parametrów
logging.info(f"Najlepsze parametry dla modelu Random Forest: {best_params}")

# Wytrenowanie modelu z najlepszymi parametrami
rf_best = grid_search.best_estimator_

# Ocena modelu na zbiorze testowym
y_test_pred_best = rf_best.predict(X_test_prepared)

# Obliczanie metryk dla zoptymalizowanego modelu
test_mse_best = mean_squared_error(y_test, y_test_pred_best)
test_mae_best = mean_absolute_error(y_test, y_test_pred_best)
test_rmse_best = np.sqrt(test_mse_best)
test_r2_best = r2_score(y_test, y_test_pred_best)
test_mape_best = mean_absolute_percentage_error(y_test, y_test_pred_best)

# Logowanie wyników po optymalizacji
logging.info(f"Po optymalizacji - Wyniki na zbiorze testowym: MSE = {test_mse_best:.4f}, MAE = {test_mae_best:.4f}, RMSE = {test_rmse_best:.4f}, R^2 = {test_r2_best:.4f}, MAPE = {test_mape_best:.4f}")
print(f"Po optymalizacji - Wyniki na zbiorze testowym: MSE = {test_mse_best:.4f}, MAE = {test_mae_best:.4f}, RMSE = {test_rmse_best:.4f}, R^2 = {test_r2_best:.4f}, MAPE = {test_mape_best:.4f}")