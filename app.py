import streamlit as st
import pandas as pd
import numpy as np

from module.curatare import incarca_date, curata_date, get_outlieri_info
from module.statistici import (get_distributie_severitate, get_accidente_pe_ora,
                                get_accidente_pe_an, get_top_state,
                                get_severitate_meteo, get_zi_noapte,
                                get_meteo_per_severitate)
from module.modele import (pregateste_date_cluster, calculeaza_inertii,
                            aplica_kmeans, pregateste_date_regresie_logistica,
                            antreneaza_regresie_logistica, calculeaza_metrici,
                            pregateste_date_ols, antreneaza_ols)
from module.grafice import (grafic_severitate_bar, grafic_severitate_pie,
                              grafic_accidente_ora, grafic_accidente_an,
                              grafic_top_state, grafic_severitate_meteo,
                              grafic_zi_noapte_bar, grafic_outlieri,
                              grafic_valori_lipsa, grafic_elbow,
                              grafic_scatter_clustere, grafic_distributie_clustere_bar,
                              grafic_distributie_clustere_pie, grafic_severitate_cluster,
                              grafic_confusion_matrix, grafic_coeficienti_logistic,
                              grafic_coeficienti_ols, grafic_reziduale,
                              grafic_distributie_reziduale, grafic_real_vs_estimat)

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

df = incarca_date()

# ============================================================
# PAGINA 1 — DATE BRUTE
# ============================================================
if pagina == "1. Date brute":
    st.title("Date brute")

    st.markdown("## a) Definirea problemei")
    st.markdown("""
    Înainte de orice prelucrare, este esențial să înțelegem **structura setului de date**.
    Setul provine din **Kaggle — US Accidents Dataset** (Moosavi et al., 2019) și conține
    accidente rutiere din SUA între 2016 și 2023.
    """)

    st.markdown("## b) Informații necesare")
    st.markdown("""
    - **Dimensiunea setului** — numărul de rânduri și coloane
    - **Tipurile de date** — numerice, categorice, boolean, datetime
    - **Valorile lipsă** — ce coloane au date incomplete
    - **Statistici de bază** — medie, minim, maxim, deviație standard
    """)

    st.markdown("## c) Metode de calcul")
    st.markdown("""
    - `pd.read_csv()` — citirea fișierului CSV
    - `df.shape`, `df.dtypes`, `df.describe()` — explorare structură
    - `df.isnull().sum()` — identificarea valorilor lipsă
    - Sampling stratificat: **70.000 rânduri per an** pentru reprezentare uniformă
    """)

    st.markdown("## d) Prezentarea rezultatelor")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Înregistrări", f"{df.shape[0]:,}")
    col2.metric("Variabile", df.shape[1])
    col3.metric("State acoperite", df['State'].nunique())
    col4.metric("Perioada", f"{df['Year'].min()}–{df['Year'].max()}")

    st.markdown("### Primele 10 rânduri")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### Tipurile de date per coloană")
    tip_df = pd.DataFrame({
        "Coloană": df.columns,
        "Tip de date": df.dtypes.values.astype(str),
        "Valori unice": df.nunique().values,
        "Exemplu": [str(df[col].dropna().iloc[0])
                    if len(df[col].dropna()) > 0 else "N/A" for col in df.columns]
    })
    st.dataframe(tip_df, use_container_width=True)

    st.markdown("### Statistici descriptive")
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown("### Valori lipsă per coloană")
    fig_lipsa, lipsa_df = grafic_valori_lipsa(df)
    st.plotly_chart(fig_lipsa, use_container_width=True)
    st.dataframe(lipsa_df, use_container_width=True)

    st.markdown("### Distribuția accidentelor pe ani")
    an_df = get_accidente_pe_an(df)
    st.plotly_chart(grafic_accidente_an(an_df), use_container_width=True)

    st.markdown("## e) Interpretarea economică")
    st.info(f"Datele acoperă **{df['State'].nunique()} state** din SUA.")
    st.warning("Coloane cu >50% valori lipsă vor fi eliminate în etapa de curățare.")
    st.success("**Severity** (1-4) este variabila cheie — distribuția sa ghidează toate analizele.")

# ============================================================
# PAGINA 2 — CURĂȚARE DATE
# ============================================================
elif pagina == "2. Curățare date":
    st.title("Curățare date")

    st.markdown("## a) Definirea problemei")
    st.markdown("""
    Datele brute conțin **valori lipsă, outlieri și variabile categorice** care nu pot fi
    folosite direct în algoritmii de machine learning. Scopul este obținerea unui set
    de date **curat, complet și pregătit** pentru analiză.
    """)

    st.markdown("## b) Informații necesare")
    st.markdown("""
    - **Coloane inutile** — cu >50% valori lipsă
    - **Valori lipsă numerice** — completate cu **media** coloanei
    - **Valori lipsă categorice** — completate cu **modul**
    - **Outlieri** — detectați prin metoda IQR
    - **Codificare** — LabelEncoder pentru variabile categorice
    - **Scalare** — StandardScaler pentru variabile numerice
    """)

    st.markdown("## c) Metode de calcul și formule")
    st.markdown("### Metoda IQR")
    st.latex(r"IQR = Q_3 - Q_1")
    st.latex(r"\text{Lower} = Q_1 - 1.5 \times IQR \quad \text{Upper} = Q_3 + 1.5 \times IQR")

    st.markdown("### StandardScaler")
    st.latex(r"z = \frac{x - \mu}{\sigma}")

    st.markdown("## d) Prezentarea rezultatelor")

    df_curat = df.copy()

    # Eliminare coloane
    st.markdown("### Eliminare coloane cu >50% valori lipsă")
    cols_eliminate = ['Precipitation(in)']
    cols_eliminate = [c for c in cols_eliminate if c in df_curat.columns]
    df_curat.drop(columns=cols_eliminate, inplace=True)
    st.info(f"Eliminate: {', '.join(cols_eliminate)}")

    # Valori lipsă numerice
    st.markdown("### Completare valori lipsă numerice cu media")
    cols_numerice = ['Temperature(C)', 'Humidity(%)', 'Pressure(in)',
                     'Visibility(mi)', 'Wind_Speed(mph)']
    cols_numerice = [c for c in cols_numerice if c in df_curat.columns]
    inainte = {col: df_curat[col].isnull().sum() for col in cols_numerice}
    for col in cols_numerice:
        df_curat[col] = df_curat[col].fillna(df_curat[col].mean())
    lipsa_num = pd.DataFrame({
        "Coloană": cols_numerice,
        "Lipsă înainte": [inainte[c] for c in cols_numerice],
        "Lipsă după": [df_curat[c].isnull().sum() for c in cols_numerice],
        "Media folosită": [f"{df_curat[c].mean():.2f}" for c in cols_numerice]
    })
    st.dataframe(lipsa_num, use_container_width=True)

    # Valori lipsă categorice
    st.markdown("### Completare valori lipsă categorice cu modul")
    cols_cat = ['City', 'Zipcode', 'Timezone', 'Wind_Direction',
                'Weather_Condition', 'Sunrise_Sunset', 'Civil_Twilight',
                'Nautical_Twilight', 'Astronomical_Twilight']
    cols_cat = [c for c in cols_cat if c in df_curat.columns]
    inainte_cat = {col: df_curat[col].isnull().sum() for col in cols_cat}
    for col in cols_cat:
        df_curat[col] = df_curat[col].fillna(df_curat[col].mode()[0])
    lipsa_cat = pd.DataFrame({
        "Coloană": cols_cat,
        "Lipsă înainte": [inainte_cat[c] for c in cols_cat],
        "Modul folosit": [df_curat[c].mode()[0] for c in cols_cat]
    })
    st.dataframe(lipsa_cat, use_container_width=True)

    # Outlieri
    st.markdown("### Detecție și eliminare outlieri (IQR)")
    n_inainte = len(df_curat)
    outlieri_info, limite_info = get_outlieri_info(df_curat)
    st.dataframe(pd.DataFrame(limite_info), use_container_width=True)
    st.plotly_chart(grafic_outlieri(outlieri_info), use_container_width=True)

    df_curat = curata_date(df)
    col1, col2, col3 = st.columns(3)
    col1.metric("Înainte", f"{n_inainte:,}")
    col2.metric("După curățare", f"{len(df_curat):,}")
    col3.metric("Eliminate", f"{n_inainte - len(df_curat):,}")

    # Codificare
    st.markdown("### Codificare (LabelEncoder)")
    cols_encode = ['State', 'Timezone', 'Wind_Direction',
                   'Weather_Condition', 'Sunrise_Sunset']
    cols_encode = [c for c in cols_encode if c in df_curat.columns]
    exemplu = pd.DataFrame({
        "Coloană": cols_encode,
        "Valori unice": [df_curat[c].nunique() for c in cols_encode],
        "Exemplu original": [df_curat[c].iloc[0] for c in cols_encode],
        "Exemplu codificat": [int(df_curat[c + '_cod'].iloc[0])
                               for c in cols_encode if c + '_cod' in df_curat.columns]
    })
    st.dataframe(exemplu, use_container_width=True)

    # Scalare
    st.markdown("### Scalare (StandardScaler)")
    cols_scale = ['Temperature(C)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)']
    cols_scale = [c for c in cols_scale if c in df_curat.columns]
    st.dataframe(
        df_curat[['Temperature(C)', 'Temperature(C)_scaled',
                  'Humidity(%)', 'Humidity(%)_scaled']].head(5).round(4),
        use_container_width=True
    )

    st.markdown("## e) Interpretarea economică")
    st.success(f"Din {n_inainte:,} înregistrări am obținut {len(df_curat):,} curate.")
    st.info("Imputarea cu media e justificată pentru variabilele meteo cu distribuție stabilă.")
    st.warning("Eliminarea outlierilor previne distorsionarea coeficienților modelelor statistice.")

# ============================================================
# PAGINA 3 — STATISTICI DESCRIPTIVE
# ============================================================
elif pagina == "3. Statistici descriptive":
    st.title("Statistici descriptive")

    st.markdown("## a) Definirea problemei")
    st.markdown("""
    Identificăm tiparele principale: **când, unde și în ce condiții** se produc accidentele.
    """)

    st.markdown("## b) Informații necesare")
    st.markdown("""
    Variabile: **Severity, Start_Time, Ora, Year, State, Weather_Condition, Sunrise_Sunset,
    Temperature(C), Humidity(%), Visibility(mi)**
    """)

    st.markdown("## c) Metode de calcul")
    st.latex(r"\bar{x}_{grup} = \frac{1}{n_{grup}} \sum_{i=1}^{n_{grup}} x_i")
    st.latex(r"f_{rel} = \frac{n_{categorie}}{N_{total}} \times 100")

    st.markdown("## d) Prezentarea rezultatelor")

    # Severitate
    st.markdown("### Distribuția severității")
    sev_df = get_distributie_severitate(df)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(grafic_severitate_bar(sev_df), use_container_width=True)
    with col2:
        st.plotly_chart(grafic_severitate_pie(sev_df), use_container_width=True)
    st.dataframe(sev_df, use_container_width=True)

    # Ora
    st.markdown("### Distribuția pe ora din zi")
    ora_df = get_accidente_pe_ora(df)
    st.plotly_chart(grafic_accidente_ora(ora_df), use_container_width=True)
    st.dataframe(ora_df, use_container_width=True)

    # An
    st.markdown("### Evoluția pe ani")
    an_df = get_accidente_pe_an(df)
    st.plotly_chart(grafic_accidente_an(an_df), use_container_width=True)

    # State
    st.markdown("### Top 15 state")
    state_df = get_top_state(df)
    st.plotly_chart(grafic_top_state(state_df), use_container_width=True)
    st.dataframe(state_df, use_container_width=True)

    # Meteo
    st.markdown("### Severitate medie pe condiții meteo")
    meteo_df = get_severitate_meteo(df)
    st.plotly_chart(grafic_severitate_meteo(meteo_df), use_container_width=True)
    st.dataframe(meteo_df, use_container_width=True)

    # Zi/Noapte
    st.markdown("### Zi vs. Noapte")
    zn_df = get_zi_noapte(df)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(grafic_zi_noapte_bar(zn_df, 'nr_accidente',
                        'Nr. accidente ziua vs. noaptea'), use_container_width=True)
    with col2:
        st.plotly_chart(grafic_zi_noapte_bar(zn_df, 'severitate_medie',
                        'Severitate medie ziua vs. noaptea'), use_container_width=True)
    st.dataframe(zn_df, use_container_width=True)

    # Meteo per severitate
    st.markdown("### Condiții meteo medii per nivel de severitate")
    st.dataframe(get_meteo_per_severitate(df), use_container_width=True)

    st.markdown("## e) Interpretarea economică")
    ora_varf = ora_df.loc[ora_df['nr_accidente'].idxmax(), 'Ora']
    state_top = state_df.iloc[0]['State']
    st.success(f"Ora de vârf: **{int(ora_varf)}:00** — trafic intens de navetă.")
    st.info(f"Statul **{state_top}** necesită atenție prioritară.")
    st.warning("Vizibilitatea redusă și ceața cresc semnificativ severitatea accidentelor.")

# ============================================================
# PAGINA 4 — CLUSTERIZARE KMEANS
# ============================================================
elif pagina == "4. Clusterizare KMeans":
    st.title("Clusterizare KMeans")

    st.markdown("## a) Definirea problemei")
    st.markdown("""
    **Segmentăm accidentele în grupuri omogene** pe baza condițiilor meteo.
    Clusterizarea este **nesupervizată** — algoritmul descoperă singur structura din date.
    """)

    st.markdown("## b) Informații necesare")
    st.markdown("Variabile: **Temperature(C), Humidity(%), Visibility(mi), Wind_Speed(mph), Distance(mi)**")

    st.markdown("## c) Metode de calcul")
    st.latex(r"J = \sum_{k=1}^{K} \sum_{x_i \in C_k} ||x_i - \mu_k||^2")
    st.latex(r"d(x_i, \mu_k) = \sqrt{\sum_{j=1}^{p} (x_{ij} - \mu_{kj})^2}")
    st.markdown("""
    **Pașii algoritmului:**
    1. Inițializare K centroizi
    2. Atribuire puncte la cel mai apropiat centroid
    3. Recalculare centroizi ca medie a clusterului
    4. Repetare până la convergență
    """)

    st.markdown("## d) Prezentarea rezultatelor")

    df_clean, X, features = pregateste_date_cluster(df)
    K_range, inertii = calculeaza_inertii(X)

    st.markdown("### Metoda cotului")
    st.plotly_chart(grafic_elbow(K_range, inertii), use_container_width=True)

    k_ales = st.slider("Alege numărul de clustere K", 2, 6, 3)
    df_cl, km = aplica_kmeans(X, df_clean, features, k_ales, df)

    # Distributie
    st.markdown("### Distribuția pe clustere")
    dist_cl = df_cl['Cluster'].value_counts().reset_index()
    dist_cl.columns = ['Cluster', 'Nr. accidente']
    dist_cl['Procent (%)'] = (dist_cl['Nr. accidente'] /
                               dist_cl['Nr. accidente'].sum() * 100).round(1)
    dist_cl = dist_cl.sort_values('Cluster')
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(grafic_distributie_clustere_bar(dist_cl), use_container_width=True)
    with col2:
        st.plotly_chart(grafic_distributie_clustere_pie(dist_cl), use_container_width=True)

    # Scatter
    st.markdown("### Vizualizare clustere")
    st.plotly_chart(grafic_scatter_clustere(
        df_cl, 'Temperature(C)', 'Humidity(%)',
        'Clustere — Temperatură vs. Umiditate',
        'Temperatură (°C)', 'Umiditate (%)'
    ), use_container_width=True)
    st.plotly_chart(grafic_scatter_clustere(
        df_cl, 'Visibility(mi)', 'Wind_Speed(mph)',
        'Clustere — Vizibilitate vs. Viteză vânt',
        'Vizibilitate (mi)', 'Viteză vânt (mph)'
    ), use_container_width=True)

    # Profil
    st.markdown("### Profilul clusterelor")
    profil = df_cl.groupby('Cluster')[features].mean().round(2)
    st.dataframe(profil, use_container_width=True)

    # Severitate per cluster
    st.markdown("### Severitatea medie per cluster")
    sev_cl = df_cl.groupby('Cluster')['Severity'].mean().round(2).reset_index()
    sev_cl.columns = ['Cluster', 'Severitate medie']
    st.plotly_chart(grafic_severitate_cluster(sev_cl), use_container_width=True)

    st.markdown("## e) Interpretarea economică")
    cluster_max = sev_cl.loc[sev_cl['Severitate medie'].idxmax(), 'Cluster']
    sev_max = sev_cl['Severitate medie'].max()
    st.success(f"Clusterul **{cluster_max}** are severitatea maximă ({sev_max:.2f}/4.0).")
    st.info("Clusterele permit alocarea diferențiată a resurselor de intervenție.")
    st.warning("KMeans presupune clustere sferice — pentru date complexe se recomandă DBSCAN.")

# ============================================================
# PAGINA 5 — REGRESIE LOGISTICĂ
# ============================================================
elif pagina == "5. Regresie logistică":
    st.title("Regresie logistică")

    st.markdown("## a) Definirea problemei")
    st.markdown("""
    Prezicem dacă un accident va fi **grav (Severity ≥ 3)** sau **negrav (Severity < 3)**.
    Este o problemă de **clasificare binară**.
    """)

    st.markdown("## b) Informații necesare")
    st.markdown("""
    **Predictori:** Temperature(C), Humidity(%), Visibility(mi), Wind_Speed(mph),
    Distance(mi), Sunrise_Sunset, Weather_Condition, Junction, Traffic_Signal, Crossing
    
    **Target:** grav = 1 dacă Severity ≥ 3, altfel 0
    """)

    st.markdown("## c) Metode de calcul")
    st.latex(r"P(grav=1) = \frac{1}{1 + e^{-z}}")
    st.latex(r"z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n")
    st.markdown("""
    - P > 0.5 → **grav** | P ≤ 0.5 → **negrav**
    - Split: 80% antrenament, 20% testare
    - Scalare: StandardScaler înainte de antrenament
    """)

    st.markdown("## d) Prezentarea rezultatelor")

    X_train, X_test, y_train, y_test, features = pregateste_date_regresie_logistica(df)
    model = antreneaza_regresie_logistica(X_train, y_train)
    metrici = calculeaza_metrici(model, X_test, y_test)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Acuratețe", f"{metrici['accuracy']:.1f}%")
    col2.metric("Precizie", f"{metrici['precision']:.1f}%")
    col3.metric("Recall", f"{metrici['recall']:.1f}%")
    col4.metric("F1 Score", f"{metrici['f1']:.1f}%")

    st.markdown("""
    **Metricile explicate:**
    - **Acuratețe** — % din toate accidentele clasificate corect
    - **Precizie** — din cele prezise grave, câte chiar erau grave
    - **Recall** — din toate gravele reale, câte a detectat modelul
    - **F1** — media armonică precizie/recall
    """)

    st.markdown("### Matricea de confuzie")
    cm = metrici['confusion_matrix']
    st.plotly_chart(grafic_confusion_matrix(cm), use_container_width=True)
    st.markdown(f"""
    - **{cm[0][0]}** negrave prezise corect ✅
    - **{cm[1][1]}** grave prezise corect ✅
    - **{cm[0][1]}** negrave prezise greșit ca grave ❌
    - **{cm[1][0]}** grave prezise greșit ca negrave ❌
    """)

    st.markdown("### Importanța variabilelor")
    st.plotly_chart(
        grafic_coeficienti_logistic(features, model.coef_[0]),
        use_container_width=True
    )
    st.markdown("""
    - **Coeficient pozitiv** → crește probabilitatea unui accident grav
    - **Coeficient negativ** → scade probabilitatea
    """)

    st.markdown("## e) Interpretarea economică")
    st.success("Modelul permite prioritizarea intervențiilor pentru accidentele cu risc ridicat.")
    st.info("Vizibilitatea și intersecțiile sunt cei mai puternici predictori ai gravității.")
    st.warning("Un sistem de alertă în timp real poate folosi modelul pentru dispatch prioritar.")

# ============================================================
# PAGINA 6 — REGRESIE OLS
# ============================================================
elif pagina == "6. Regresie OLS":
    st.title("Regresie multiplă OLS")

    st.markdown("## a) Definirea problemei")
    st.markdown("""
    Modelăm **severitatea accidentului** ca variabilă continuă în funcție de condițiile
    meteo și infrastructură, folosind metoda celor mai mici pătrate (OLS).
    """)

    st.markdown("## b) Informații necesare")
    st.markdown("""
    **y (dependent):** Severity (1-4)
    
    **x (predictori):** Temperature(C), Humidity(%), Visibility(mi),
    Wind_Speed(mph), Distance(mi), Junction, Traffic_Signal
    """)

    st.markdown("## c) Metode de calcul")
    st.latex(r"Severity = \beta_0 + \beta_1 Temp + \beta_2 Humidity + "
             r"\beta_3 Visibility + \ldots + \varepsilon")
    st.latex(r"SSR = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2")
    st.latex(r"\hat{\beta} = (X^T X)^{-1} X^T y")
    st.latex(r"R^2 = 1 - \frac{SSR}{SST}")

    st.markdown("## d) Prezentarea rezultatelor")

    X_ols, y_ols, features_ols, df_sample = pregateste_date_ols(df)
    model_ols = antreneaza_ols(X_ols, y_ols)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", f"{model_ols.rsquared:.4f}")
    col2.metric("R² ajustat", f"{model_ols.rsquared_adj:.4f}")
    col3.metric("F-statistic", f"{model_ols.fvalue:.2f}")
    col4.metric("Nr. observații", f"{int(model_ols.nobs):,}")

    st.markdown(f"**R² = {model_ols.rsquared:.4f}** — modelul explică "
                f"{model_ols.rsquared*100:.1f}% din varianța severității.")

    st.markdown("### Coeficienții modelului")
    coef_df = pd.DataFrame({
        "Variabilă": model_ols.params.index,
        "Coeficient": model_ols.params.values.round(4),
        "Eroare standard": model_ols.bse.values.round(4),
        "t-statistic": model_ols.tvalues.values.round(3),
        "p-value": model_ols.pvalues.values.round(4),
        "Semnificativ": ["✅" if p < 0.05 else "❌" for p in model_ols.pvalues.values]
    })
    st.dataframe(coef_df, use_container_width=True)
    st.plotly_chart(grafic_coeficienti_ols(coef_df), use_container_width=True)

    st.markdown("### Analiza reziduelor")
    st.warning("""
    **Notă metodologică:** Severity are doar 4 valori discrete (1-4), ceea ce produce
    benzi orizontale în graficul reziduelor. Aceasta e o limitare a OLS pentru variabile
    discrete — pentru modelare mai precisă se recomandă regresia logistică ordinală.
    """)
    fitted = model_ols.fittedvalues
    residuals = model_ols.resid
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(grafic_reziduale(fitted, residuals), use_container_width=True)
    with col2:
        st.plotly_chart(grafic_distributie_reziduale(residuals), use_container_width=True)

    st.markdown("### Valori reale vs. estimate")
    st.plotly_chart(
        grafic_real_vs_estimat(y_ols.values, fitted.values),
        use_container_width=True
    )

    st.markdown("## e) Interpretarea economică")
    coef_viz = model_ols.params.get('Visibility(mi)', 0)
    coef_jct = model_ols.params.get('Junction', 0)
    st.success(f"Vizibilitatea: coeficient **{coef_viz:.4f}** — fiecare milă în plus "
               f"scade severitatea cu {abs(coef_viz):.4f} puncte.")
    st.info(f"Intersecțiile: coeficient **{coef_jct:.4f}** — accidentele la intersecții "
            f"sunt {'mai grave' if coef_jct > 0 else 'mai puțin grave'}.")
    st.warning(f"R² mic ({model_ols.rsquared:.4f}) sugerează factori nemăsurați importanți "
               f"(viteza vehiculului, tipul impactului).")

# ============================================================
# PAGINA 7 — CONCLUZII
# ============================================================
elif pagina == "7. Concluzii":
    st.title("Concluzii și recomandări")

    st.markdown("## a) Definirea problemei")
    st.markdown("""
    Sintetizăm toate rezultatele obținute și formulăm recomandări concrete pentru
    **reducerea numărului și gravității accidentelor rutiere** în SUA.
    """)

    st.markdown("## b) Sinteza informațiilor")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Înregistrări", f"{df.shape[0]:,}")
    col2.metric("Variabile", df.shape[1])
    col3.metric("Ani acoperiți", f"{df['Year'].max() - df['Year'].min() + 1}")
    col4.metric("State", df['State'].nunique())

    st.markdown("""
    | Etapă | Metodă | Scop |
    |---|---|---|
    | Import | `pd.read_csv()`, sampling stratificat | Reprezentare uniformă 2016-2023 |
    | Curățare | IQR, fillna, LabelEncoder, StandardScaler | Date curate pentru modelare |
    | Statistici | groupby, agg, value_counts | Înțelegerea distribuției |
    | Clusterizare | KMeans, metoda cotului | Segmentarea accidentelor |
    | Clasificare | Regresie logistică | Predicția gravității |
    | Regresie | OLS (statsmodels) | Cuantificarea factorilor de risc |
    """)

    st.markdown("## c) Sinteza metodelor")
    st.markdown("""
    **Librării Python folosite:**
    - **Pandas** — import, curățare, grupare, agregare
    - **Scikit-learn** — LabelEncoder, StandardScaler, KMeans, LogisticRegression
    - **Statsmodels** — regresie OLS cu inferență statistică completă
    - **Plotly** — vizualizări interactive
    - **Streamlit** — interfața web
    """)

    st.markdown("## d) Rezultatele principale")

    ora_df = get_accidente_pe_ora(df)
    ora_varf = ora_df.loc[ora_df['nr_accidente'].idxmax(), 'Ora']
    state_top = get_top_state(df).iloc[0]['State']
    pct_sev2 = (df['Severity'] == 2).mean() * 100

    col1, col2 = st.columns(2)
    with col1:
        sev_df = get_distributie_severitate(df)
        st.plotly_chart(grafic_severitate_pie(sev_df), use_container_width=True)
    with col2:
        an_df = get_accidente_pe_an(df)
        st.plotly_chart(grafic_accidente_an(an_df), use_container_width=True)

    factori = pd.DataFrame({
        "Factor de risc": ["Vizibilitate redusă", "Orele de vârf (7-9, 16-18)",
                           "Intersecții", "Condiții meteo nefavorabile",
                           "Accidente nocturne"],
        "Impact": ["Ridicat", "Ridicat", "Moderat", "Moderat", "Moderat"],
        "Prioritate": ["🔴 Urgentă", "🔴 Urgentă", "🟡 Ridicată",
                       "🟡 Ridicată", "🟢 Medie"]
    })
    st.dataframe(factori, use_container_width=True)

    st.markdown("## e) Interpretarea economică și recomandări")
    st.success(f"{pct_sev2:.1f}% din accidente au severitate moderată (nivel 2) — "
               "intervențiile preventive pot reduce semnificativ costurile sociale.")

    st.markdown(f"""
    **Recomandări principale:**
    - Patrule suplimentare la ora **{int(ora_varf)}:00** — vârful accidentelor
    - Prioritizare geografică: statul **{state_top}** necesită atenție maximă
    - Sisteme de avertizare meteo în timp real pe autostrăzi
    - Reproiectarea intersecțiilor cu risc ridicat
    - Extinderea iluminatului stradal pentru reducerea accidentelor nocturne
    
    **Posibilități de extindere:**
    - Random Forest / XGBoost pentru acuratețe mai mare
    - Hartă interactivă cu zonele de risc (coordonate GPS)
    - Analiză de serii temporale și predicții pentru anii următori
    """)

    st.info("""
    **Notă metodologică:** Analiza se bazează pe 70.000 accidente per an din totalul
    de ~7.7 milioane. Rezultatele sunt reprezentative dar pot varia față de setul complet.
    """)
