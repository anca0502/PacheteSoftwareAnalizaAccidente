import pandas as pd
import numpy as np


def get_distributie_severitate(df):
    """Returneaza distributia accidentelor pe nivel de severitate."""
    sev = df['Severity'].value_counts().reset_index()
    sev.columns = ['Severitate', 'Nr. accidente']
    sev['Severitate'] = sev['Severitate'].map({
        1: '1 - Minor', 2: '2 - Moderat',
        3: '3 - Grav', 4: '4 - Foarte grav'
    })
    sev['Procent (%)'] = (sev['Nr. accidente'] /
                          sev['Nr. accidente'].sum() * 100).round(1)
    return sev.sort_values('Severitate')


def get_accidente_pe_ora(df):
    """Returneaza numarul si severitatea medie a accidentelor pe ora."""
    return df.groupby('Ora').agg(
        nr_accidente=('Severity', 'count'),
        severitate_medie=('Severity', 'mean')
    ).round(2).reset_index()


def get_accidente_pe_an(df):
    """Returneaza numarul de accidente per an."""
    return df.groupby('Year').agg(
        nr_accidente=('Severity', 'count'),
        severitate_medie=('Severity', 'mean')
    ).round(2).reset_index()


def get_top_state(df, top_n=15):
    """Returneaza top N state dupa numarul de accidente."""
    return df.groupby('State').agg(
        nr_accidente=('Severity', 'count'),
        severitate_medie=('Severity', 'mean')
    ).round(2).reset_index().sort_values(
        'nr_accidente', ascending=False
    ).head(top_n)


def get_severitate_meteo(df, min_accidente=100):
    """Returneaza severitatea medie pe conditii meteo."""
    meteo = df.groupby('Weather_Condition').agg(
        nr_accidente=('Severity', 'count'),
        severitate_medie=('Severity', 'mean')
    ).round(2).reset_index()
    return meteo[meteo['nr_accidente'] > min_accidente].sort_values(
        'severitate_medie', ascending=False
    ).head(15)


def get_zi_noapte(df):
    """Returneaza statistici zi vs noapte."""
    return df.groupby('Sunrise_Sunset').agg(
        nr_accidente=('Severity', 'count'),
        severitate_medie=('Severity', 'mean'),
        procent=('Severity', lambda x: round(len(x) / len(df) * 100, 1))
    ).round(2).reset_index()


def get_meteo_per_severitate(df):
    """Returneaza conditiile meteo medii per nivel de severitate."""
    cols = ['Temperature(C)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)']
    cols = [c for c in cols if c in df.columns]
    meteo_sev = df.groupby('Severity')[cols].mean().round(2).reset_index()
    meteo_sev['Severity'] = meteo_sev['Severity'].map({
        1: '1 - Minor', 2: '2 - Moderat',
        3: '3 - Grav', 4: '4 - Foarte grav'
    })
    return meteo_sev
