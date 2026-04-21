import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
import statsmodels.api as sm


def pregateste_date_cluster(df):
    """Pregateste datele pentru clusterizare KMeans."""
    df_cl = df.copy()
    cols_fill = ['Temperature(C)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)']
    for col in cols_fill:
        if col in df_cl.columns:
            df_cl[col] = df_cl[col].fillna(df_cl[col].mean())

    features = ['Temperature(C)', 'Humidity(%)',
                'Visibility(mi)', 'Wind_Speed(mph)', 'Distance(mi)']
    features = [f for f in features if f in df_cl.columns]

    df_clean = df_cl[features].dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(df_clean)
    return df_clean, X, features


def calculeaza_inertii(X, k_max=8):
    """Calculeaza inertiile pentru metoda cotului."""
    inertii = []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertii.append(km.inertia_)
    return list(range(2, k_max + 1)), inertii


def aplica_kmeans(X, df_clean, features, k, df_original):
    """Aplica KMeans si returneaza dataframe cu clustere si severitate."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_result = df_clean.copy()
    df_result['Cluster'] = km.fit_predict(X).astype(str)

    # Adaugam severitatea din df original
    df_result['Severity'] = df_original['Severity'].values[:len(df_result)]
    return df_result, km


def pregateste_date_regresie_logistica(df):
    """Pregateste datele pentru regresia logistica."""
    df_rl = df.copy()
    cols_fill = ['Temperature(C)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)']
    for col in cols_fill:
        if col in df_rl.columns:
            df_rl[col] = df_rl[col].fillna(df_rl[col].mean())

    df_rl['Sunrise_Sunset_cod'] = LabelEncoder().fit_transform(
        df_rl['Sunrise_Sunset'].fillna('Day')
    )
    df_rl['Weather_cod'] = LabelEncoder().fit_transform(
        df_rl['Weather_Condition'].fillna('Clear')
    )
    df_rl['grav'] = (df_rl['Severity'] >= 3).astype(int)

    features = ['Temperature(C)', 'Humidity(%)', 'Visibility(mi)',
                'Wind_Speed(mph)', 'Distance(mi)',
                'Sunrise_Sunset_cod', 'Weather_cod',
                'Junction', 'Traffic_Signal', 'Crossing']
    features = [f for f in features if f in df_rl.columns]

    df_clean = df_rl[features + ['grav']].dropna()
    n = min(5000, len(df_clean))
    df_sample = df_clean.sample(n, random_state=42)

    X = df_sample[features].astype(float)
    y = df_sample['grav']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, features


def antreneaza_regresie_logistica(X_train, y_train):
    """Antreneaza modelul de regresie logistica."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def calculeaza_metrici(model, X_test, y_test):
    """Calculeaza metricile de performanta."""
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred) * 100,
        'precision': precision_score(y_test, y_pred, zero_division=0) * 100,
        'recall': recall_score(y_test, y_pred, zero_division=0) * 100,
        'f1': f1_score(y_test, y_pred, zero_division=0) * 100,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_pred': y_pred
    }


def pregateste_date_ols(df):
    """Pregateste datele pentru regresia OLS."""
    df_ols = df.copy()
    cols_fill = ['Temperature(C)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)']
    for col in cols_fill:
        if col in df_ols.columns:
            df_ols[col] = df_ols[col].fillna(df_ols[col].mean())

    features_ols = ['Temperature(C)', 'Humidity(%)', 'Visibility(mi)',
                    'Wind_Speed(mph)', 'Distance(mi)', 'Junction', 'Traffic_Signal']
    features_ols = [f for f in features_ols if f in df_ols.columns]

    df_clean = df_ols[features_ols + ['Severity']].dropna()
    n = min(50000, len(df_clean))
    df_sample = df_clean.sample(n, random_state=42)

    X_ols = sm.add_constant(df_sample[features_ols].astype(float))
    y_ols = df_sample['Severity']
    return X_ols, y_ols, features_ols, df_sample


def antreneaza_ols(X_ols, y_ols):
    """Antreneaza modelul OLS."""
    return sm.OLS(y_ols, X_ols).fit()
