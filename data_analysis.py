import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

# Konfiguracja loggera
logging.basicConfig(
    filename='data_analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

# Wczytanie datasetu
df = pd.read_csv('./CollegeDistance.csv')

# Wstępna analiza danych
logging.info("Wstępna analiza danych.")
logging.info(f"Liczba wierszy: {df.shape[0]}")
logging.info(f"Liczba kolumn: {df.shape[1]}")
logging.info(f"Kolumny: {df.columns.tolist()}")

# Sprawdzanie brakujących wartości
missing_values = df.isnull().sum()
logging.info("\nBrakujące wartości w każdej kolumnie:")
logging.info(f"\n{missing_values[missing_values > 0]}")

# Tworzenie folderu img, jeśli nie istnieje
if not os.path.exists('img'):
    os.makedirs('img')
    logging.info("Folder 'img' został utworzony.")
else:
    logging.info("Folder 'img' już istnieje.")

# # Podział zmiennych na numeryczne i kategoryczne
num_cols = ['unemp', 'wage', 'distance', 'tuition', 'education']
cat_cols = ['gender', 'ethnicity', 'fcollege', 'mcollege', 'home', 'urban', 'income', 'region']

# Wykresy dla zmiennych numerycznych (zależności od score)
for col in num_cols:
    try:
        plt.figure(figsize=(10, 6))
        df['binned_' + col] = pd.qcut(df[col], q=5, duplicates='drop')
        sns.boxplot(x='binned_' + col, y='score', data=df)
        plt.title(f'{col} vs. score')
        plt.xlabel(col)
        plt.ylabel('score')
        plt.tight_layout()
        plt.savefig(f'img/{col}_vs_score.png')
        plt.close()
        logging.info(f"Wykres zależności {col} vs. score został zapisany jako img/{col}_vs_score.png")
    except Exception as e:
        logging.error(f"Błąd przy tworzeniu wykresu {col} vs. score: {e}")

# Wykresy dla zmiennych kategorycznych (zależności od score)
for col in cat_cols:
    try:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=col, y='score', data=df)
        plt.title(f'{col} vs. score')
        plt.xlabel(col)
        plt.ylabel('score')
        plt.tight_layout()
        plt.savefig(f'img/{col}_vs_score.png')
        plt.close()
        logging.info(f"Wykres zależności {col} vs. score został zapisany jako img/{col}_vs_score.png")
    except Exception as e:
        logging.error(f"Błąd przy tworzeniu wykresu {col} vs. score: {e}")

# Wykresy rozkładu dla zmiennych numerycznych
for col in num_cols:
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True)
        plt.axvline(df[col].mean(), color='r', linestyle='--', label='Średnia')
        plt.axvline(df[col].median(), color='g', linestyle='-', label='Mediana')
        plt.legend()
        plt.title(f'Rozkład {col}')
        plt.xlabel(col)
        plt.ylabel('Liczba obserwacji')
        plt.tight_layout()
        plt.savefig(f'img/{col}_distribution.png')
        plt.close()
        logging.info(f"Wykres rozkładu {col} został zapisany jako img/{col}_distribution.png")
    except Exception as e:
        logging.error(f"Błąd przy tworzeniu wykresu rozkładu {col}: {e}")

# Wykresy rozkładu dla zmiennych kategorycznych
for col in cat_cols:
    try:
        plt.figure(figsize=(10, 6))
        category_distribution = df[col].value_counts(normalize=True) * 100
        sns.barplot(x=category_distribution.index, y=category_distribution.values)
        plt.title(f'Rozkład {col}')
        plt.xlabel(col)
        plt.ylabel('Procentowy udział')
        plt.tight_layout()
        plt.savefig(f'img/{col}_distribution.png')
        plt.close()
        logging.info(f"Wykres rozkładu {col} został zapisany jako img/{col}_distribution.png")
    except Exception as e:
        logging.error(f"Błąd przy tworzeniu wykresu rozkładu {col}: {e}")