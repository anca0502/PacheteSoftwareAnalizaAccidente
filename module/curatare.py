import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st


@st.cache_data
def incarca_date():
    df_full = pd.read_csv("Data/US_Accidents_March23.csv",
                          usecols=['Severity', 'Start_Time', 'State', 'City',
                                   'County', 'Zipcode', 'Timezone',
                                   'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
                                   'Visibility(mi)', 'Wind_Direction', 'Wind_Speed(mph)',
                                   'Precipitation(in)', 'Weather_Condition',
                                   'Amenity', 'Bump', 'Crossing', 'Junction',
                                   'Traffic_Signal', 'Sunrise_Sunset',
                                   'Civil_Twilight', 'Nautical_Twilight',
                                   'Astronomical_Twilight', 'Distance(mi)'])

    df_full['Start_Time'] = pd.to_datetime(df_full['Start_Time'], format='mixed')
    df_full['Year'] = df_full['Start_Time'].dt.year
    df_full['Ora'] = df_full['Start_Time'].dt.hour
    df_full['Temperature(C)'] = ((df_full['Temperature(F)'] - 32) * 5 / 9).round(1)

    df_sample = df_full.groupby('Year', group_keys=False).apply(
        lambda x: x.sample(min(len(x), 70000), random_state=42)
    ).reset_index(drop=True)

    return df_sample


def curata_date(df):
    """Returneaza un dataframe curatat complet."""
    df_curat = df.copy()

    # Eliminare coloane inutile
    cols_eliminate = ['Precipitation(in)']
    cols_eliminate = [c for c in cols_eliminate if c in df_curat.columns]
    df_curat.drop(columns=cols_eliminate, inplace=True)

    # Completare valori lipsa numerice cu media
    cols_numerice = ['Temperature(C)', 'Humidity(%)', 'Pressure(in)',
                     'Visibility(mi)', 'Wind_Speed(mph)']
    cols_numerice = [c for c in cols_numerice if c in df_curat.columns]
    for col in cols_numerice:
        df_curat[col] = df_curat[col].fillna(df_curat[col].mean())

    # Completare valori lipsa categorice cu modul
    cols_categorice = ['City', 'Zipcode', 'Timezone', 'Wind_Direction',
                       'Weather_Condition', 'Sunrise_Sunset',
                       'Civil_Twilight', 'Nautical_Twilight',
                       'Astronomical_Twilight']
    cols_categorice = [c for c in cols_categorice if c in df_curat.columns]
    for col in cols_categorice:
        df_curat[col] = df_curat[col].fillna(df_curat[col].mode()[0])

    # Eliminare outlieri prin IQR
    cols_outlieri = ['Temperature(C)', 'Humidity(%)',
                     'Visibility(mi)', 'Wind_Speed(mph)', 'Distance(mi)']
    cols_outlieri = [c for c in cols_outlieri if c in df_curat.columns]
    for col in cols_outlieri:
        Q1 = df_curat[col].quantile(0.25)
        Q3 = df_curat[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_curat = df_curat[
            (df_curat[col] >= lower) & (df_curat[col] <= upper)
        ]

    # Codificare variabile categorice
    cols_encode = ['State', 'Timezone', 'Wind_Direction',
                   'Weather_Condition', 'Sunrise_Sunset']
    cols_encode = [c for c in cols_encode if c in df_curat.columns]
    le = LabelEncoder()
    for col in cols_encode:
        df_curat[col + '_cod'] = le.fit_transform(df_curat[col].astype(str))

    # Scalare variabile numerice
    cols_scale = ['Temperature(C)', 'Humidity(%)',
                  'Visibility(mi)', 'Wind_Speed(mph)']
    cols_scale = [c for c in cols_scale if c in df_curat.columns]
    scaler = StandardScaler()
    df_curat[[c + '_scaled' for c in cols_scale]] = scaler.fit_transform(
        df_curat[cols_scale]
    )

    return df_curat


def get_outlieri_info(df):
    """Returneaza un dict cu numarul de outlieri per coloana."""
    cols_outlieri = ['Temperature(C)', 'Humidity(%)',
                     'Visibility(mi)', 'Wind_Speed(mph)', 'Distance(mi)']
    cols_outlieri = [c for c in cols_outlieri if c in df.columns]
    outlieri_info = {}
    limite_info = []
    for col in cols_outlieri:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        outlieri_info[col] = int(n_out)
        limite_info.append({
            "Coloană": col, "Q1": round(Q1, 2), "Q3": round(Q3, 2),
            "IQR": round(IQR, 2), "Lower bound": round(lower, 2),
            "Upper bound": round(upper, 2), "Outlieri găsiți": int(n_out)
        })
    return outlieri_info, limite_info
