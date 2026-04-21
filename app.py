import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# CONFIGURARE
# ============================================================
st.set_page_config(
    page_title="Analiza Accidentelor Rutiere SUA",
    page_icon="🚗",
    layout="wide"
)

st.sidebar.title("🚗 Accidente Rutiere SUA")
pagina = st.sidebar.radio("Navigare", [
    "1. Date brute",
    "2. Curățare date",
    "3. Statistici descriptive",
    "4. Clusterizare KMeans",
    "5. Regresie logistică",
    "6. Regresie OLS",
    "7. Concluzii"
])

# ============================================================
# IMPORT DATE
# ============================================================
@st.cache_data
def incarca_date():
    # Citim tot fisierul dar doar coloanele necesare
    df_full = pd.read_csv("Data/US_Accidents_March23.csv")

    # Extragem anul
    df_full['Year'] = pd.to_datetime(df_full['Start_Time'], format='mixed').dt.year
    # Conversie Fahrenheit → Celsius
    df_full['Temperature(C)'] = ((df_full['Temperature(F)'] - 32) * 5 / 9).round(1)
    df['Ora'] = pd.to_datetime(df['Start_Time'], format='mixed').dt.hour

    # Luam 70.000 randuri din fiecare an
    df_sample = df_full.groupby('Year', group_keys=False).apply(
        lambda x: x.sample(min(len(x), 700), random_state=42)
    ).reset_index(drop=True)

    return df_sample


df = incarca_date()


# ============================================================
# PAGINA 1 — DATE BRUTE
# ============================================================
if pagina == "1. Date brute":
    st.title("Date brute")
    st.markdown("**Definiția problemei:** Înțelegem structura setului de date înainte de orice prelucrare.")
    st.markdown("**Sursa:** Kaggle — US Accidents Dataset (Moosavi et al., 2019)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Înregistrări", f"{df.shape[0]:,}")
    col2.metric("Variabile", df.shape[1])
    col3.metric("State acoperite", df['State'].nunique())
    col4.metric("Perioada", f"{df['Start_Time'].min()[:4]}–{df['Start_Time'].max()[:4]}")

    st.subheader("Primele 10 rânduri")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Statistici descriptive")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Valori lipsă per coloană")
    lipsa = df.isnull().sum().reset_index()
    lipsa.columns = ["Coloană", "Valori lipsă"]
    lipsa["Procent (%)"] = (lipsa["Valori lipsă"] / len(df) * 100).round(2)
    lipsa = lipsa[lipsa["Valori lipsă"] > 0].sort_values("Valori lipsă", ascending=False)
    st.dataframe(lipsa, use_container_width=True)

# ============================================================
# PAGINA 2 — CURĂȚARE DATE
# ============================================================
elif pagina == "2. Curățare date":
    st.title("Curățare date")
    st.markdown("**Definiția problemei:** Pregătim datele pentru analiză prin eliminarea coloanelor inutile, tratarea valorilor lipsă și a outlierilor.")

    df_curat = df.copy()

    # --- Eliminare coloane cu prea multe valori lipsă ---
    st.subheader("Eliminare coloane cu >50% valori lipsă")
    cols_eliminate = ['End_Lat', 'End_Lng', 'Wind_Chill(F)',
                      'Precipitation(in)', 'Description',
                      'Weather_Timestamp', 'Airport_Code']
    df_curat.drop(columns=cols_eliminate, inplace=True)
    st.info(f"Eliminate {len(cols_eliminate)} coloane: {', '.join(cols_eliminate)}")

    # --- Completare valori lipsă numerice cu media ---
    st.subheader("Completare valori lipsă numerice cu media")
    cols_numerice = ['Temperature(C)', 'Humidity(%)', 'Pressure(in)',
                     'Visibility(mi)', 'Wind_Speed(mph)']
    for col in cols_numerice:
        media = df_curat[col].mean()
        df_curat[col].fillna(media, inplace=True)
    st.success(f"Completate cu media: {', '.join(cols_numerice)}")

    # --- Completare valori lipsă categorice cu modul ---
    cols_categorice = ['City', 'Zipcode', 'Timezone', 'Wind_Direction',
                       'Weather_Condition', 'Sunrise_Sunset',
                       'Civil_Twilight', 'Nautical_Twilight',
                       'Astronomical_Twilight']
    for col in cols_categorice:
        df_curat[col].fillna(df_curat[col].mode()[0], inplace=True)
    st.success(f"Completate cu modul: {', '.join(cols_categorice)}")

    # --- Outlieri prin IQR ---
    st.subheader("Detecție și eliminare outlieri (metoda IQR)")
    cols_outlieri = ['Temperature(C)', 'Humidity(%)',
                     'Visibility(mi)', 'Wind_Speed(mph)', 'Distance(mi)']
    n_inainte = len(df_curat)
    outlieri_info = {}
    for col in cols_outlieri:
        Q1 = df_curat[col].quantile(0.25)
        Q3 = df_curat[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_out = ((df_curat[col] < lower) | (df_curat[col] > upper)).sum()
        outlieri_info[col] = int(n_out)
        df_curat = df_curat[
            (df_curat[col] >= lower) & (df_curat[col] <= upper)
        ]

    fig_out = px.bar(
        x=list(outlieri_info.keys()),
        y=list(outlieri_info.values()),
        labels={"x": "Coloană", "y": "Nr. outlieri eliminați"},
        title="Outlieri detectați per variabilă",
        color=list(outlieri_info.values()),
        color_continuous_scale="Reds"
    )
    st.plotly_chart(fig_out, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Înainte", f"{n_inainte:,}")
    col2.metric("După curățare", f"{len(df_curat):,}")
    col3.metric("Eliminate", f"{n_inainte - len(df_curat):,}")

    # --- Codificare ---
    st.subheader("Codificare variabile categorice (LabelEncoder)")
    cols_encode = ['State', 'Timezone', 'Wind_Direction',
                   'Weather_Condition', 'Sunrise_Sunset']
    le = LabelEncoder()
    for col in cols_encode:
        df_curat[col + '_cod'] = le.fit_transform(df_curat[col].astype(str))
    st.success(f"Codificate: {', '.join(cols_encode)}")

    # --- Scalare ---
    st.subheader("Scalare variabile numerice (StandardScaler)")
    cols_scale = ['Temperature(C)', 'Humidity(%)',
                  'Visibility(mi)', 'Wind_Speed(mph)']
    scaler = StandardScaler()
    df_curat[[c + '_scaled' for c in cols_scale]] = scaler.fit_transform(
        df_curat[cols_scale]
    )
    st.success("Scalare completă — valorile au acum medie 0 și deviație standard 1.")
    st.dataframe(df_curat[['Temperature(C)', 'Temperature(C)_scaled',
                             'Humidity(%)', 'Humidity(%)_scaled']].head(5),
                 use_container_width=True)

# ============================================================
# PAGINA 3 — STATISTICI DESCRIPTIVE
# ============================================================
elif pagina == "3. Statistici descriptive":
    st.title("Statistici descriptive")
    st.markdown("**Definiția problemei:** Identificăm tiparele principale din date — când, unde și în ce condiții se produc accidentele.")

    # Distribuție severitate
    sev = df['Severity'].value_counts().reset_index()
    sev.columns = ['Severitate', 'Nr. accidente']
    sev['Severitate'] = sev['Severitate'].map({
        1: '1 - Minor', 2: '2 - Moderat',
        3: '3 - Grav', 4: '4 - Foarte grav'
    })
    fig1 = px.bar(sev, x='Severitate', y='Nr. accidente',
                  title='Distribuția accidentelor pe nivel de severitate',
                  color='Nr. accidente', color_continuous_scale='Reds')
    st.plotly_chart(fig1, use_container_width=True)

    # Accidente pe oră
    df['Ora'] = pd.to_datetime(df['Start_Time'], format='mixed').dt.hour
    ora_grp = df.groupby('Ora').size().reset_index(name='Nr. accidente')
    fig2 = px.line(ora_grp, x='Ora', y='Nr. accidente',
                   title='Distribuția accidentelor pe ora din zi',
                   markers=True)
    st.plotly_chart(fig2, use_container_width=True)

    # Top state
    state_grp = df.groupby('State').size().reset_index(name='Nr. accidente')
    state_grp = state_grp.sort_values('Nr. accidente', ascending=False).head(15)
    fig3 = px.bar(state_grp, x='State', y='Nr. accidente',
                  title='Top 15 state cu cele mai multe accidente',
                  color='Nr. accidente', color_continuous_scale='Blues')
    st.plotly_chart(fig3, use_container_width=True)

    # Severitate medie pe condiții meteo
    st.subheader("Severitate medie pe condiții meteo")
    meteo_grp = df.groupby('Weather_Condition').agg(
        nr_accidente=('Severity', 'count'),
        severitate_medie=('Severity', 'mean')
    ).round(2).reset_index()
    meteo_grp = meteo_grp[meteo_grp['nr_accidente'] > 500].sort_values(
        'severitate_medie', ascending=False
    ).head(15)
    fig4 = px.bar(meteo_grp, x='Weather_Condition', y='severitate_medie',
                  title='Severitate medie pe condiții meteo (min. 500 accidente)',
                  color='severitate_medie', color_continuous_scale='Oranges')
    st.plotly_chart(fig4, use_container_width=True)

    # Grupare zi/noapte
    st.subheader("Accidente ziua vs. noaptea")
    zi_noapte = df.groupby('Sunrise_Sunset').agg(
        nr=('Severity', 'count'),
        sev_medie=('Severity', 'mean')
    ).round(2).reset_index()
    st.dataframe(zi_noapte, use_container_width=True)

# ============================================================
# PAGINA 4 — CLUSTERIZARE
# ============================================================
elif pagina == "4. Clusterizare KMeans":
    st.title("Clusterizare KMeans")
    st.markdown("**Definiția problemei:** Segmentăm accidentele în grupuri omogene pe baza condițiilor meteo și a severității.")
    st.markdown("**Formula:** KMeans minimizează suma distanțelor euclidiene față de centroizi: `Σ ||xi - μk||²`")

    df_cl = df.copy()
    cols_fill = ['Temperature(C)', 'Humidity(%)',
                 'Visibility(mi)', 'Wind_Speed(mph)']
    for col in cols_fill:
        df_cl[col].fillna(df_cl[col].mean(), inplace=True)

    features = ['Temperature(C)', 'Humidity(%)',
                'Visibility(mi)', 'Wind_Speed(mph)', 'Distance(mi)']
    scaler = StandardScaler()

    # Folosim sample pentru viteză
    df_sample = df_cl[features].dropna().sample(50000, random_state=42)
    X = scaler.fit_transform(df_sample)

    # Metoda cotului
    st.subheader("Metoda cotului — alegerea K optim")
    inertii = []
    K_range = range(2, 9)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertii.append(km.inertia_)

    fig_elbow = px.line(x=list(K_range), y=inertii,
                        labels={"x": "K (nr. clustere)", "y": "Inerție"},
                        title="Metoda cotului", markers=True)
    st.plotly_chart(fig_elbow, use_container_width=True)

    k_ales = st.slider("Alege numărul de clustere", 2, 6, 3)
    km_final = KMeans(n_clusters=k_ales, random_state=42, n_init=10)
    df_sample_idx = df_cl[features].dropna().sample(50000, random_state=42)
    df_sample_idx['Cluster'] = km_final.fit_predict(X)

    fig_scatter = px.scatter(
        df_sample_idx, x='Temperature(C)', y='Humidity(%)',
        color=df_sample_idx['Cluster'].astype(str),
        title='Clustere accidente (Temperatură vs. Umiditate)',
        labels={'color': 'Cluster'},
        opacity=0.4
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Profilul clusterelor")
    profil = df_sample_idx.groupby('Cluster')[features].mean().round(2)
    st.dataframe(profil, use_container_width=True)

    st.markdown("**Interpretare economică:** Clusterele relevă tipare distincte de condiții în care se produc accidentele — ex. accidente pe ploaie vs. senin vs. ceață.")

# ============================================================
# PAGINA 5 — REGRESIE LOGISTICĂ
# ============================================================
elif pagina == "5. Regresie logistică":
    st.title("Regresie logistică")
    st.markdown("**Definiția problemei:** Predicția dacă un accident va fi grav (Severity ≥ 3) sau nu.")
    st.markdown("**Formula:** `P(y=1) = 1 / (1 + e^(-z))` unde `z = β₀ + β₁x₁ + ... + βₙxₙ`")

    df_rl = df.copy()
    cols_fill = ['Temperature(C)', 'Humidity(%)',
                 'Visibility(mi)', 'Wind_Speed(mph)']
    for col in cols_fill:
        df_rl[col].fillna(df_rl[col].mean(), inplace=True)

    df_rl['Sunrise_Sunset_cod'] = LabelEncoder().fit_transform(
        df_rl['Sunrise_Sunset'].fillna('Day')
    )
    df_rl['Weather_cod'] = LabelEncoder().fit_transform(
        df_rl['Weather_Condition'].fillna('Clear')
    )

    # Target binar: grav (3-4) vs. neGrav (1-2)
    df_rl['grav'] = (df_rl['Severity'] >= 3).astype(int)

    features = ['Temperature(C)', 'Humidity(%)', 'Visibility(mi)',
                'Wind_Speed(mph)', 'Distance(mi)',
                'Sunrise_Sunset_cod', 'Weather_cod',
                'Junction', 'Traffic_Signal', 'Crossing']

    df_rl_clean = df_rl[features + ['grav']].dropna()
    df_sample = df_rl_clean.sample(100000, random_state=42)

    X = df_sample[features].astype(float)
    y = df_sample['grav']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Acuratețe", f"{accuracy_score(y_test, y_pred)*100:.1f}%")
    col2.metric("Precizie", f"{precision_score(y_test, y_pred)*100:.1f}%")
    col3.metric("Recall", f"{recall_score(y_test, y_pred)*100:.1f}%")
    col4.metric("F1 Score", f"{f1_score(y_test, y_pred)*100:.1f}%")

    # Matrice confuzie
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, text_auto=True,
                       labels=dict(x="Predicție", y="Real"),
                       x=['Neгrav', 'Grav'], y=['Negrav', 'Grav'],
                       title="Matricea de confuzie",
                       color_continuous_scale="Blues")
    st.plotly_chart(fig_cm, use_container_width=True)

    # Coeficienți
    coef_df = pd.DataFrame({
        "Variabilă": features,
        "Coeficient": model.coef_[0]
    }).sort_values("Coeficient")
    fig_coef = px.bar(coef_df, x="Coeficient", y="Variabilă",
                      orientation="h",
                      title="Importanța variabilelor",
                      color="Coeficient",
                      color_continuous_scale="RdBu")
    st.plotly_chart(fig_coef, use_container_width=True)

    st.markdown("**Interpretare economică:** Vizibilitatea redusă și prezența unei intersecții cresc semnificativ probabilitatea unui accident grav.")

# ============================================================
# PAGINA 6 — REGRESIE OLS
# ============================================================
elif pagina == "6. Regresie OLS":
    st.title("Regresie multiplă OLS")
    st.markdown("**Definiția problemei:** Modelăm severitatea accidentului în funcție de condițiile meteo și de infrastructură.")
    st.markdown("**Formula:** `Severity = β₀ + β₁·Temp + β₂·Humidity + β₃·Visibility + ... + ε`")

    df_ols = df.copy()
    cols_fill = ['Temperature(C)', 'Humidity(%)',
                 'Visibility(mi)', 'Wind_Speed(mph)']
    for col in cols_fill:
        df_ols[col].fillna(df_ols[col].mean(), inplace=True)

    features_ols = ['Temperature(C)', 'Humidity(%)',
                    'Visibility(mi)', 'Wind_Speed(mph)',
                    'Distance(mi)', 'Junction', 'Traffic_Signal']

    df_ols_clean = df_ols[features_ols + ['Severity']].dropna()
    df_sample = df_ols_clean.sample(50000, random_state=42)

    X_ols = sm.add_c