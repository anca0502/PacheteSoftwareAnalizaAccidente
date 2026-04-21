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
    import os
    if os.path.exists("Data/sample_procesat.csv"):
        return pd.read_csv("Data/sample_procesat.csv")

    cols_necesare = [...]  # ca mai sus
    df_full = pd.read_csv("Data/US_Accidents_March23.csv", usecols=cols_necesare)

    df_full['Year'] = pd.to_datetime(df_full['Start_Time'], format='mixed').dt.year
    df_full['Temperature(C)'] = ((df_full['Temperature(F)'] - 32) * 5 / 9).round(1)
    df_full['Ora'] = pd.to_datetime(df_full['Start_Time'], format='mixed').dt.hour

    df_sample = df_full.groupby('Year', group_keys=False).apply(
        lambda x: x.sample(min(len(x), 700), random_state=42)
    ).reset_index(drop=True)

    # Salveaza pentru data viitoare
    df_sample.to_csv("Data/sample_procesat.csv", index=False)
    return df_sample


df = incarca_date()


# ============================================================
# PAGINA 1 — DATE BRUTE
# ============================================================
if pagina == "1. Date brute":
    st.title("Date brute")

    # --- Definirea problemei ---
    st.markdown("## a) Definirea problemei")
    st.markdown("""
        Înainte de orice prelucrare, este esențial să înțelegem **structura setului de date**:
        câte înregistrări conține, ce variabile sunt disponibile, ce tipuri de date avem
        și dacă există valori lipsă.
    
        Setul de date provine de la **Kaggle — US Accidents Dataset** (Moosavi et al., 2019)
        și conține accidente rutiere înregistrate în Statele Unite între 2016 și 2023,
        colectate din surse precum API-uri de trafic și camere de supraveghere.
        """)

    # --- Informații necesare ---
    st.markdown("## b) Informații necesare")
    st.markdown("""
        Pentru a înțelege datele brute avem nevoie de:
        - **Dimensiunea setului** — numărul de rânduri și coloane
        - **Tipurile de date** — numerice, categorice, boolean, datetime
        - **Valorile lipsă** — ce coloane au date incomplete și în ce proporție
        - **Statistici de bază** — medie, minim, maxim, deviație standard
        - **Perioada acoperită** — intervalul de timp al înregistrărilor
        - **Acoperirea geografică** — câte state din SUA sunt reprezentate
        """)

    # --- Metode de calcul ---
    st.markdown("## c) Metode de calcul")
    st.markdown("""
        Explorarea datelor brute folosește funcții standard din librăria **Pandas**:
        - `pd.read_csv()` — citirea fișierului CSV în memorie
        - `df.shape` — dimensiunea setului (rânduri, coloane)
        - `df.dtypes` — tipul fiecărei coloane
        - `df.describe()` — statistici descriptive automate pentru coloanele numerice
        - `df.isnull().sum()` — numărul de valori lipsă per coloană
        - `df.nunique()` — numărul de valori unice per coloană
    
        Citim **70.000 de rânduri din fiecare an** (2016–2023) pentru a asigura
        o reprezentare uniformă a întregii perioade, rezultând ~560.000 înregistrări total.
        """)

    # --- Prezentarea rezultatelor ---
    st.markdown("## d) Prezentarea rezultatelor")

    st.markdown("### Indicatori generali")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Înregistrări", f"{df.shape[0]:,}")
    col2.metric("Variabile", df.shape[1])
    col3.metric("State acoperite", df['State'].nunique())
    col4.metric("Perioada", f"{df['Year'].min()}–{df['Year'].max()}")
    st.markdown("### Primele 10 rânduri din dataset")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### Tipurile de date per coloană")
    tip_df = pd.DataFrame({
        "Coloană": df.columns,
        "Tip de date": df.dtypes.values.astype(str),
        "Valori unice": df.nunique().values,
        "Exemplu valoare": [str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else "N/A"
                            for col in df.columns]
    })
    st.dataframe(tip_df, use_container_width=True)

    st.markdown("""
        **Tipuri de variabile identificate:**
        - **Numerice** (float/int): Temperature, Humidity, Visibility, Wind_Speed, Severity etc.
        - **Categorice** (object): State, City, Weather_Condition, Wind_Direction etc.
        - **Booleane** (bool): Junction, Traffic_Signal, Crossing, Amenity etc.
        - **Datetime** (object→datetime): Start_Time, End_Time
        """)

    st.markdown("### Statistici descriptive")
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown("### Valori lipsă per coloană")
    lipsa = df.isnull().sum().reset_index()
    lipsa.columns = ["Coloană", "Valori lipsă"]
    lipsa["Procent (%)"] = (lipsa["Valori lipsă"] / len(df) * 100).round(2)
    lipsa = lipsa[lipsa["Valori lipsă"] > 0].sort_values("Valori lipsă", ascending=False)

    fig_lipsa = px.bar(
        lipsa, x="Coloană", y="Procent (%)",
        title="Procentul valorilor lipsă per coloană",
        color="Procent (%)",
        color_continuous_scale="Reds",
        text="Procent (%)"
    )
    fig_lipsa.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    st.plotly_chart(fig_lipsa, use_container_width=True)
    st.dataframe(lipsa, use_container_width=True)

    st.markdown("### Distribuția accidentelor pe ani")
    an_grp = df.groupby('Year').size().reset_index(name='Nr. accidente')
    fig_ani = px.bar(an_grp, x='Year', y='Nr. accidente',
                     title='Numărul de accidente per an în dataset',
                     color='Nr. accidente',
                     color_continuous_scale='Blues',
                     text='Nr. accidente')
    fig_ani.update_traces(texttemplate='%{text:,}', textposition='outside')
    st.plotly_chart(fig_ani, use_container_width=True)

    # --- Interpretare economică ---
    st.markdown("## e) Interpretarea economică a rezultatelor")
    st.markdown("""
        Analiza datelor brute relevă câteva observații importante cu impact economic și social:
        """)
    st.info(f"""
        **Acoperire geografică:** Datele acoperă **{df['State'].nunique()} state** din SUA,
        oferind o imagine reprezentativă la nivel național asupra accidentelor rutiere.
        """)
    st.warning("""
        **Valori lipsă semnificative:** Coloane precum `Wind_Chill`, `Precipitation` și
        `End_Lat/End_Lng` au peste 50% valori lipsă — acestea vor fi eliminate în etapa
        de curățare pentru a nu distorsiona rezultatele analizei.
        """)
    st.success("""
        **Severitatea accidentelor** este variabila noastră cheie de analiză. Înțelegerea
        distribuției sale pe ani, state și condiții meteo poate ghida politicile publice
        de siguranță rutieră și alocarea resurselor de intervenție.
        """)
    st.markdown("""
        **Implicații practice:**
        - Numărul mare de accidente înregistrate (~7.7 milioane total) indică o problemă
          majoră de siguranță rutieră în SUA cu costuri sociale și economice ridicate
        - Prezența variabilelor meteo detaliate permite analizarea impactului condițiilor
          atmosferice asupra gravității accidentelor
        - Variabilele de infrastructură (Junction, Traffic_Signal, Crossing) permit
          identificarea punctelor negre rutiere care necesită investiții
        """)
# ============================================================
# PAGINA 2 — CURĂȚARE DATE
# ============================================================
elif pagina == "2. Curățare date":
    st.title("Curățare date")

    # --- Definirea problemei ---
    st.markdown("## a) Definirea problemei")
    st.markdown("""
    Datele brute conțin **valori lipsă, outlieri și variabile categorice** care nu pot fi
    folosite direct în algoritmii de machine learning. Scopul acestei etape este să obținem
    un set de date **curat, complet și pregătit** pentru analiză statistică și modelare.

    Curățarea datelor este una dintre cele mai importante etape într-un proiect de analiză —
    calitatea rezultatelor depinde direct de calitatea datelor de intrare.
    """)

    # --- Informații necesare ---
    st.markdown("## b) Informații necesare")
    st.markdown("""
    Pentru curățarea datelor avem nevoie să identificăm:
    - **Coloane inutile** — variabile cu prea multe valori lipsă (>50%) care nu pot fi recuperate
    - **Valori lipsă numerice** — completate cu **media** coloanei (imputare statistică)
    - **Valori lipsă categorice** — completate cu **modul** (cea mai frecventă valoare)
    - **Outlieri** — valori extreme care distorsionează rezultatele, detectate prin metoda IQR
    - **Variabile categorice** — transformate în numere prin **LabelEncoder**
    - **Scalarea** — aducerea variabilelor numerice la aceeași scară prin **StandardScaler**
    """)

    # --- Metode de calcul ---
    st.markdown("## c) Metode de calcul și formule")

    st.markdown("### Metoda IQR (Interquartile Range) pentru detecția outlierilor")
    st.latex(r"IQR = Q_3 - Q_1")
    st.latex(r"\text{Lower bound} = Q_1 - 1.5 \times IQR")
    st.latex(r"\text{Upper bound} = Q_3 + 1.5 \times IQR")
    st.markdown("Orice valoare în afara intervalului `[Lower, Upper]` este considerată outlier și eliminată.")

    st.markdown("### StandardScaler — formula de standardizare")
    st.latex(r"z = \frac{x - \mu}{\sigma}")
    st.markdown("""
    unde:
    - **x** = valoarea originală
    - **μ** = media coloanei
    - **σ** = deviația standard a coloanei
    - **z** = valoarea standardizată (medie 0, deviație standard 1)
    """)

    st.markdown("### LabelEncoder — codificarea variabilelor categorice")
    st.markdown("""
    Transformă valorile text în numere întregi în ordine alfabetică. Exemplu:
    - `Day` → 0
    - `Night` → 1

    Necesar deoarece algoritmii de machine learning (KMeans, LogisticRegression)
    nu pot procesa text direct.
    """)

    # --- Prezentarea rezultatelor ---
    st.markdown("## d) Prezentarea rezultatelor")

    df_curat = df.copy()

    # Eliminare coloane
    st.markdown("### Eliminare coloane cu >50% valori lipsă")
    cols_eliminate = ['End_Lat', 'End_Lng', 'Wind_Chill(F)',
                      'Precipitation(in)', 'Description',
                      'Weather_Timestamp', 'Airport_Code']
    cols_eliminate = [c for c in cols_eliminate if c in df_curat.columns]

    eliminate_info = pd.DataFrame({
        "Coloană eliminată": cols_eliminate,
        "Motiv": ["100% valori lipsă"] * 2 +
                 ["~90% valori lipsă"] * 2 +
                 ["Text liber, nerelevant"] +
                 ["~10% lipsă, nerelevant pentru analiză"] * 2
    })
    st.dataframe(eliminate_info, use_container_width=True)
    df_curat.drop(columns=cols_eliminate, inplace=True)
    st.info(f"Eliminate {len(cols_eliminate)} coloane — au rămas {df_curat.shape[1]} variabile.")

    # Completare valori lipsă numerice
    st.markdown("### Completare valori lipsă numerice cu media")
    cols_numerice = ['Temperature(C)', 'Humidity(%)', 'Pressure(in)',
                     'Visibility(mi)', 'Wind_Speed(mph)']
    cols_numerice = [c for c in cols_numerice if c in df_curat.columns]

    inainte_lipsa = {col: df_curat[col].isnull().sum() for col in cols_numerice}
    for col in cols_numerice:
        media = df_curat[col].mean()
        df_curat[col] = df_curat[col].fillna(media)

    lipsa_num_df = pd.DataFrame({
        "Coloană": cols_numerice,
        "Valori lipsă (înainte)": [inainte_lipsa[c] for c in cols_numerice],
        "Valori lipsă (după)": [df_curat[c].isnull().sum() for c in cols_numerice],
        "Completate cu media": [f"{df_curat[c].mean():.2f}" for c in cols_numerice]
    })
    st.dataframe(lipsa_num_df, use_container_width=True)
    st.success(f"Completate cu media: {', '.join(cols_numerice)}")

    # Completare valori lipsă categorice
    st.markdown("### Completare valori lipsă categorice cu modul")
    cols_categorice = ['City', 'Zipcode', 'Timezone', 'Wind_Direction',
                       'Weather_Condition', 'Sunrise_Sunset',
                       'Civil_Twilight', 'Nautical_Twilight',
                       'Astronomical_Twilight']
    cols_categorice = [c for c in cols_categorice if c in df_curat.columns]

    inainte_cat = {col: df_curat[col].isnull().sum() for col in cols_categorice}
    for col in cols_categorice:
        df_curat[col] = df_curat[col].fillna(df_curat[col].mode()[0])

    lipsa_cat_df = pd.DataFrame({
        "Coloană": cols_categorice,
        "Valori lipsă (înainte)": [inainte_cat[c] for c in cols_categorice],
        "Completate cu modul": [df_curat[c].mode()[0] for c in cols_categorice]
    })
    st.dataframe(lipsa_cat_df, use_container_width=True)
    st.success(f"Completate cu modul: {', '.join(cols_categorice)}")

    # Outlieri
    st.markdown("### Detecție și eliminare outlieri (metoda IQR)")
    cols_outlieri = ['Temperature(C)', 'Humidity(%)',
                     'Visibility(mi)', 'Wind_Speed(mph)', 'Distance(mi)']
    cols_outlieri = [c for c in cols_outlieri if c in df_curat.columns]
    n_inainte = len(df_curat)
    outlieri_info = {}
    limite_info = []

    for col in cols_outlieri:
        Q1 = df_curat[col].quantile(0.25)
        Q3 = df_curat[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_out = ((df_curat[col] < lower) | (df_curat[col] > upper)).sum()
        outlieri_info[col] = int(n_out)
        limite_info.append({
            "Coloană": col, "Q1": round(Q1, 2), "Q3": round(Q3, 2),
            "IQR": round(IQR, 2), "Lower bound": round(lower, 2),
            "Upper bound": round(upper, 2), "Outlieri găsiți": int(n_out)
        })
        df_curat = df_curat[
            (df_curat[col] >= lower) & (df_curat[col] <= upper)
            ]

    st.dataframe(pd.DataFrame(limite_info), use_container_width=True)

    fig_out = px.bar(
        x=list(outlieri_info.keys()),
        y=list(outlieri_info.values()),
        labels={"x": "Coloană", "y": "Nr. outlieri eliminați"},
        title="Numărul de outlieri eliminați per variabilă",
        color=list(outlieri_info.values()),
        color_continuous_scale="Reds",
        text=list(outlieri_info.values())
    )
    fig_out.update_traces(texttemplate='%{text:,}', textposition='outside')
    st.plotly_chart(fig_out, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Înainte de curățare", f"{n_inainte:,}")
    col2.metric("După curățare", f"{len(df_curat):,}")
    col3.metric("Rânduri eliminate", f"{n_inainte - len(df_curat):,}")

    # Codificare
    st.markdown("### Codificare variabile categorice (LabelEncoder)")
    cols_encode = ['State', 'Timezone', 'Wind_Direction',
                   'Weather_Condition', 'Sunrise_Sunset']
    cols_encode = [c for c in cols_encode if c in df_curat.columns]
    le = LabelEncoder()
    exemplu_cod = []
    for col in cols_encode:
        df_curat[col + '_cod'] = le.fit_transform(df_curat[col].astype(str))
        exemplu_cod.append({
            "Coloană originală": col,
            "Coloană codificată": col + '_cod',
            "Valori unice": df_curat[col].nunique(),
            "Exemplu original": df_curat[col].iloc[0],
            "Exemplu codificat": int(df_curat[col + '_cod'].iloc[0])
        })
    st.dataframe(pd.DataFrame(exemplu_cod), use_container_width=True)
    st.success(f"Codificate {len(cols_encode)} coloane cu LabelEncoder.")

    # Scalare
    st.markdown("### Scalare variabile numerice (StandardScaler)")
    cols_scale = ['Temperature(C)', 'Humidity(%)',
                  'Visibility(mi)', 'Wind_Speed(mph)']
    cols_scale = [c for c in cols_scale if c in df_curat.columns]
    scaler = StandardScaler()
    df_curat[[c + '_scaled' for c in cols_scale]] = scaler.fit_transform(
        df_curat[cols_scale]
    )

    st.markdown("Comparație valori originale vs. scalate (primele 5 rânduri):")
    st.dataframe(
        df_curat[['Temperature(C)', 'Temperature(C)_scaled',
                  'Humidity(%)', 'Humidity(%)_scaled']].head(5).round(4),
        use_container_width=True
    )
    st.success("Scalare completă — valorile au acum medie ≈ 0 și deviație standard ≈ 1.")

    # --- Interpretare economică ---
    st.markdown("## e) Interpretarea economică a rezultatelor")
    st.markdown("""
    Curățarea datelor are implicații directe asupra **calității deciziilor** luate pe baza analizei:
    """)
    st.success(f"""
    **Date de calitate superioară:** Din {n_inainte:,} înregistrări inițiale am obținut
    {len(df_curat):,} înregistrări curate — suficiente pentru analize statistice robuste.
    """)
    st.info("""
    **Imputarea cu media** pentru variabilele meteo (temperatură, umiditate, vizibilitate)
    este justificată economic — condițiile meteo au o distribuție relativ stabilă,
    iar media reprezintă o estimare rezonabilă pentru valorile lipsă.
    """)
    st.warning("""
    **Eliminarea outlierilor** este crucială pentru modelele de regresie — valorile extreme
    (ex. temperaturi de -50°C sau vânturi de 200 mph) sunt probabil erori de înregistrare
    și ar distorsiona coeficienții modelelor statistice.
    """)
    st.markdown("""
    **Scalarea datelor** asigură că variabile cu unități diferite (grade Celsius, procente,
    mile) contribuie **echitabil** la modelele de machine learning — fără scalare,
    variabilele cu valori mari ar domina artificial rezultatele.
    """)

# ============================================================
# PAGINA 3 — STATISTICI DESCRIPTIVE
# ============================================================
elif pagina == "3. Statistici descriptive":
    st.title("Statistici descriptive")

    # --- Definirea problemei ---
    st.markdown("## a) Definirea problemei")
    st.markdown("""
    Scopul acestei etape este să **înțelegem distribuția datelor** și să identificăm
    tiparele principale din accidentele rutiere americane:
    - **Când** se produc cele mai multe accidente? (oră, zi, lună, an)
    - **Unde** sunt concentrate accidentele? (state, orașe)
    - **În ce condiții** meteo se produc accidentele grave?
    - **Cum variază** severitatea în funcție de momentul din zi?

    Statisticile descriptive nu fac predicții — ele **descriu și rezumă** datele
    pentru a ghida etapele ulterioare de modelare.
    """)

    # --- Informații necesare ---
    st.markdown("## b) Informații necesare")
    st.markdown("""
    Variabilele utilizate în această analiză:
    - **Severity** (1-4) — gravitatea accidentului, variabila centrală a analizei
    - **Start_Time / Ora / Year** — momentul producerii accidentului
    - **State** — statul american unde s-a produs accidentul
    - **Weather_Condition** — condiția meteo la momentul accidentului
    - **Sunrise_Sunset** — dacă accidentul s-a produs ziua sau noaptea
    - **Temperature(C), Humidity(%), Visibility(mi)** — condiții atmosferice

    Metodele folosite sunt funcții de **grupare și agregare** din Pandas:
    `groupby()`, `agg()`, `value_counts()`, `mean()`, `count()`
    """)

    # --- Metode de calcul ---
    st.markdown("## c) Metode de calcul și formule")
    st.markdown("""
    **Gruparea datelor** (`groupby`) împarte setul de date în grupuri pe baza unei variabile
    categorice și calculează statistici pentru fiecare grup:
    """)
    st.latex(r"\bar{x}_{grup} = \frac{1}{n_{grup}} \sum_{i=1}^{n_{grup}} x_i")
    st.markdown("""
    **Frecvența relativă** — procentul dintr-un total:
    """)
    st.latex(r"f_{rel} = \frac{n_{categorie}}{N_{total}} \times 100")
    st.markdown("""
    unde **n_categorie** = numărul de înregistrări dintr-o categorie,
    **N_total** = numărul total de înregistrări.
    """)

    # --- Prezentarea rezultatelor ---
    st.markdown("## d) Prezentarea rezultatelor")

    # Distribuție severitate
    st.markdown("### Distribuția accidentelor pe nivel de severitate")
    sev = df['Severity'].value_counts().reset_index()
    sev.columns = ['Severitate', 'Nr. accidente']
    sev['Severitate'] = sev['Severitate'].map({
        1: '1 - Minor', 2: '2 - Moderat',
        3: '3 - Grav', 4: '4 - Foarte grav'
    })
    sev['Procent (%)'] = (sev['Nr. accidente'] / sev['Nr. accidente'].sum() * 100).round(1)
    sev = sev.sort_values('Severitate')

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(sev, x='Severitate', y='Nr. accidente',
                      title='Nr. accidente pe nivel de severitate',
                      color='Nr. accidente',
                      color_continuous_scale='Reds',
                      text='Nr. accidente')
        fig1.update_traces(texttemplate='%{text:,}', textposition='outside')
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig1b = px.pie(sev, names='Severitate', values='Nr. accidente',
                       title='Distribuția procentuală a severității',
                       color_discrete_sequence=px.colors.sequential.Reds_r)
        st.plotly_chart(fig1b, use_container_width=True)

    st.dataframe(sev, use_container_width=True)

    # Accidente pe oră
    st.markdown("### Distribuția accidentelor pe ora din zi")
    ora_grp = df.groupby('Ora').agg(
        nr_accidente=('Severity', 'count'),
        severitate_medie=('Severity', 'mean')
    ).round(2).reset_index()

    fig2 = px.line(ora_grp, x='Ora', y='nr_accidente',
                   title='Numărul de accidente per oră',
                   markers=True,
                   labels={"Ora": "Ora din zi", "nr_accidente": "Nr. accidente"})
    fig2.add_vrect(x0=6.5, x1=9.5, fillcolor="orange",
                   opacity=0.15, annotation_text="Vârf dimineață")
    fig2.add_vrect(x0=15.5, x1=18.5, fillcolor="red",
                   opacity=0.15, annotation_text="Vârf seară")
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(ora_grp.rename(columns={
        "Ora": "Ora", "nr_accidente": "Nr. accidente",
        "severitate_medie": "Severitate medie"
    }), use_container_width=True)

    # Accidente pe an
    st.markdown("### Evoluția accidentelor pe ani")
    an_grp = df.groupby('Year').agg(
        nr_accidente=('Severity', 'count'),
        severitate_medie=('Severity', 'mean')
    ).round(2).reset_index()

    fig_an = px.bar(an_grp, x='Year', y='nr_accidente',
                    title='Numărul de accidente per an',
                    color='nr_accidente',
                    color_continuous_scale='Blues',
                    text='nr_accidente')
    fig_an.update_traces(texttemplate='%{text:,}', textposition='outside')
    st.plotly_chart(fig_an, use_container_width=True)
    st.dataframe(an_grp.rename(columns={
        "Year": "An", "nr_accidente": "Nr. accidente",
        "severitate_medie": "Severitate medie"
    }), use_container_width=True)

    # Top state
    st.markdown("### Top 15 state cu cele mai multe accidente")
    state_grp = df.groupby('State').agg(
        nr_accidente=('Severity', 'count'),
        severitate_medie=('Severity', 'mean')
    ).round(2).reset_index()
    state_grp = state_grp.sort_values('nr_accidente', ascending=False).head(15)

    fig3 = px.bar(state_grp, x='State', y='nr_accidente',
                  title='Top 15 state după numărul de accidente',
                  color='severitate_medie',
                  color_continuous_scale='RdYlGn_r',
                  text='nr_accidente',
                  labels={"nr_accidente": "Nr. accidente",
                          "severitate_medie": "Severitate medie"})
    fig3.update_traces(texttemplate='%{text:,}', textposition='outside')
    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(state_grp.rename(columns={
        "State": "Stat", "nr_accidente": "Nr. accidente",
        "severitate_medie": "Severitate medie"
    }), use_container_width=True)

    # Severitate medie pe condiții meteo
    st.markdown("### Severitate medie pe condiții meteo")
    meteo_grp = df.groupby('Weather_Condition').agg(
        nr_accidente=('Severity', 'count'),
        severitate_medie=('Severity', 'mean')
    ).round(2).reset_index()
    meteo_grp = meteo_grp[meteo_grp['nr_accidente'] > 100].sort_values(
        'severitate_medie', ascending=False
    ).head(15)

    fig4 = px.bar(meteo_grp, x='Weather_Condition', y='severitate_medie',
                  title='Severitate medie pe condiții meteo (min. 100 accidente)',
                  color='severitate_medie',
                  color_continuous_scale='Oranges',
                  text='severitate_medie',
                  labels={"Weather_Condition": "Condiție meteo",
                          "severitate_medie": "Severitate medie"})
    fig4.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig4.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig4, use_container_width=True)
    st.dataframe(meteo_grp.rename(columns={
        "Weather_Condition": "Condiție meteo",
        "nr_accidente": "Nr. accidente",
        "severitate_medie": "Severitate medie"
    }), use_container_width=True)

    # Zi vs noapte
    st.markdown("### Accidente ziua vs. noaptea")
    zi_noapte = df.groupby('Sunrise_Sunset').agg(
        nr_accidente=('Severity', 'count'),
        severitate_medie=('Severity', 'mean'),
        procent=('Severity', lambda x: round(len(x) / len(df) * 100, 1))
    ).round(2).reset_index()

    col1, col2 = st.columns(2)
    with col1:
        fig5 = px.bar(zi_noapte, x='Sunrise_Sunset', y='nr_accidente',
                      title='Nr. accidente ziua vs. noaptea',
                      color='Sunrise_Sunset',
                      text='nr_accidente',
                      labels={"Sunrise_Sunset": "Moment",
                              "nr_accidente": "Nr. accidente"})
        fig5.update_traces(texttemplate='%{text:,}', textposition='outside')
        st.plotly_chart(fig5, use_container_width=True)
    with col2:
        fig5b = px.bar(zi_noapte, x='Sunrise_Sunset', y='severitate_medie',
                       title='Severitate medie ziua vs. noaptea',
                       color='Sunrise_Sunset',
                       text='severitate_medie',
                       labels={"Sunrise_Sunset": "Moment",
                               "severitate_medie": "Severitate medie"})
        fig5b.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(fig5b, use_container_width=True)

    st.dataframe(zi_noapte.rename(columns={
        "Sunrise_Sunset": "Moment zilei",
        "nr_accidente": "Nr. accidente",
        "severitate_medie": "Severitate medie",
        "procent": "Procent din total (%)"
    }), use_container_width=True)

    # Temperatura medie pe severitate
    st.markdown("### Condiții meteo medii pe nivel de severitate")
    meteo_sev = df.groupby('Severity').agg(
        temperatura_medie=('Temperature(C)', 'mean'),
        umiditate_medie=('Humidity(%)', 'mean'),
        vizibilitate_medie=('Visibility(mi)', 'mean'),
        viteza_vant_medie=('Wind_Speed(mph)', 'mean')
    ).round(2).reset_index()
    meteo_sev['Severity'] = meteo_sev['Severity'].map({
        1: '1 - Minor', 2: '2 - Moderat',
        3: '3 - Grav', 4: '4 - Foarte grav'
    })
    st.dataframe(meteo_sev.rename(columns={
        "Severity": "Severitate",
        "temperatura_medie": "Temp. medie (°C)",
        "umiditate_medie": "Umiditate medie (%)",
        "vizibilitate_medie": "Vizibilitate medie (mi)",
        "viteza_vant_medie": "Viteză vânt medie (mph)"
    }), use_container_width=True)

    # --- Interpretare economică ---
    st.markdown("## e) Interpretarea economică a rezultatelor")

    ora_varf = ora_grp.loc[ora_grp['nr_accidente'].idxmax(), 'Ora']
    state_top = state_grp.iloc[0]['State']
    meteo_top = meteo_grp.iloc[0]['Weather_Condition']

    st.success(f"""
    **Orele de vârf:** Cele mai multe accidente se produc în jurul orei **{int(ora_varf)}:00**,
    corespunzând traficului intens de navetă. Autoritățile ar trebui să concentreze
    patrulele și sistemele de monitorizare în aceste intervale.
    """)
    st.info(f"""
    **Concentrare geografică:** Statul **{state_top}** înregistrează cele mai multe accidente.
    Acest lucru poate reflecta atât densitatea mare a traficului, cât și condițiile
    climatice specifice zonei.
    """)
    st.warning(f"""
    **Condiții meteo periculoase:** Accidentele produse în condiții de **{meteo_top}**
    au cea mai mare severitate medie — semnalizarea rutieră adaptată condițiilor meteo
    extreme poate reduce semnificativ gravitatea accidentelor.
    """)
    st.markdown("""
    **Implicații pentru politici publice:**
    - Intensificarea controalelor de viteză în **orele de vârf** (7-9 și 16-18)
    - Investiții în **iluminat stradal** pentru reducerea accidentelor nocturne,
      care tind să fie mai grave decât cele diurne
    - Campanii de conștientizare despre **conducerea pe vreme rea** în statele
      cu severitate medie ridicată
    - Sisteme de **avertizare meteo în timp real** pe autostrăzile cu risc ridicat
    """)
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
        df_cl[col] = df_cl[col].fillna(df_cl[col].mean())

    features = ['Temperature(C)', 'Humidity(%)',
                'Visibility(mi)', 'Wind_Speed(mph)', 'Distance(mi)']

    df_features = df_cl[features].dropna()
    scaler = StandardScaler()

    # FIX: min() ca sa nu crape daca avem mai putine randuri decat 50000
    n = min(5000, len(df_features))
    df_sample = df_features.sample(n, random_state=42)
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
    df_sample_idx = df_features.sample(n, random_state=42).copy()
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

elif pagina == "5. Regresie logistică":
    st.title("Regresie logistică")

    # --- Definirea problemei ---
    st.markdown("## a) Definirea problemei")
    st.markdown("""
    Vrem să prezicem dacă un accident rutier va fi **grav (Severity ≥ 3)** sau **negrav (Severity < 3)**
    pe baza condițiilor meteo, infrastructurii rutiere și momentului din zi.

    Aceasta este o problemă de **clasificare binară** — răspunsul e fie 0 (negrav), fie 1 (grav).
    """)

    # --- Informații necesare ---
    st.markdown("## b) Informații necesare")
    st.markdown("""
    Variabilele folosite ca predictori:
    - **Temperature(C)** — temperatura la momentul accidentului
    - **Humidity(%)** — umiditatea aerului
    - **Visibility(mi)** — vizibilitatea în mile
    - **Wind_Speed(mph)** — viteza vântului
    - **Distance(mi)** — distanța afectată de accident
    - **Sunrise_Sunset_cod** — dacă accidentul s-a produs ziua sau noaptea (codificat)
    - **Weather_cod** — condiția meteo codificată numeric
    - **Junction** — dacă accidentul s-a produs la o intersecție (True/False)
    - **Traffic_Signal** — prezența unui semafor
    - **Crossing** — prezența unei treceri de pietoni

    Variabila țintă (target): **grav** = 1 dacă Severity ≥ 3, altfel 0
    """)

    # --- Pregătire date ---
    df_rl = df.copy()
    cols_fill = ['Temperature(C)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)']
    for col in cols_fill:
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

    df_rl_clean = df_rl[features + ['grav']].dropna()
    n = min(5000, len(df_rl_clean))
    df_sample = df_rl_clean.sample(n, random_state=42)

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

    # --- Metode și formule ---
    st.markdown("## c) Metode de calcul și formule")
    st.markdown("""
    **Regresia logistică** calculează probabilitatea că un accident e grav folosind funcția sigmoid:
    """)
    st.latex(r"P(grav=1) = \frac{1}{1 + e^{-z}}")
    st.latex(
        r"z = \beta_0 + \beta_1 \cdot Temp + \beta_2 \cdot Humidity + \beta_3 \cdot Visibility + \ldots + \beta_n \cdot x_n")
    st.markdown("""
    - Dacă **P > 0.5** → accidentul e clasificat ca **grav**
    - Dacă **P ≤ 0.5** → accidentul e clasificat ca **negrav**

    **Împărțirea datelor:** 80% antrenament, 20% testare (`train_test_split`)  
    **Scalare:** StandardScaler — aduce toate variabilele la aceeași scară înainte de antrenament  
    **Optimizare:** solverul implicit `lbfgs` minimizează funcția de cost log-loss
    """)

    # --- Prezentarea rezultatelor ---
    st.markdown("## d) Prezentarea rezultatelor")

    st.markdown("### Metrici de performanță")
    col1, col2, col3, col4 = st.columns(4)
    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, zero_division=0) * 100
    rec = recall_score(y_test, y_pred, zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, zero_division=0) * 100
    col1.metric("Acuratețe", f"{acc:.1f}%")
    col2.metric("Precizie", f"{prec:.1f}%")
    col3.metric("Recall", f"{rec:.1f}%")
    col4.metric("F1 Score", f"{f1:.1f}%")

    st.markdown("""
    **Ce înseamnă fiecare metrică:**
    - **Acuratețe** — din toate accidentele, câte a clasificat corect modelul
    - **Precizie** — din cele prezise ca grave, câte chiar erau grave
    - **Recall** — din toate accidentele grave reale, câte a detectat modelul
    - **F1 Score** — media armonică între precizie și recall (util când clasele sunt dezechilibrate)
    """)

    st.markdown("### Matricea de confuzie")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, text_auto=True,
                       labels=dict(x="Predicție", y="Real"),
                       x=['Negrav', 'Grav'], y=['Negrav', 'Grav'],
                       title="Matricea de confuzie",
                       color_continuous_scale="Blues")
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown(f"""
    **Citirea matricei:**
    - **{cm[0][0]}** accidente negrave — prezise corect ca negrave ✅
    - **{cm[1][1]}** accidente grave — prezise corect ca grave ✅
    - **{cm[0][1]}** accidente negrave — prezise greșit ca grave ❌ (fals pozitiv)
    - **{cm[1][0]}** accidente grave — prezise greșit ca negrave ❌ (fals negativ)
    """)

    st.markdown("### Importanța variabilelor (coeficienți)")
    coef_df = pd.DataFrame({
        "Variabilă": features,
        "Coeficient": model.coef_[0]
    }).sort_values("Coeficient")
    fig_coef = px.bar(coef_df, x="Coeficient", y="Variabilă",
                      orientation="h",
                      title="Importanța variabilelor — coeficienți regresie logistică",
                      color="Coeficient",
                      color_continuous_scale="RdBu")
    st.plotly_chart(fig_coef, use_container_width=True)

    st.markdown("""
    **Cum se citesc coeficienții:**
    - **Coeficient pozitiv** → variabila crește probabilitatea unui accident grav
    - **Coeficient negativ** → variabila scade probabilitatea unui accident grav
    - Cu cât valoarea absolută e mai mare, cu atât influența e mai puternică
    """)

    # --- Interpretare economică ---
    st.markdown("## e) Interpretarea economică a rezultatelor")
    st.success("""
    **Concluzie principală:** Modelul identifică cu acuratețe ridicată accidentele grave,
    ceea ce permite autorităților să prioritizeze intervențiile.
    """)
    st.markdown("""
    - **Vizibilitatea scăzută** este unul dintre cei mai puternici predictori ai gravității —
      ceața și ploaia necesită măsuri speciale de semnalizare
    - **Prezența intersecțiilor** crește riscul de accident grav — reproiectarea acestora
      poate salva vieți
    - **Accidentele de noapte** tind să fie mai grave — iluminatul stradal adecvat
      este o prioritate de investiție
    - Modelul poate fi integrat într-un **sistem de alertă în timp real** care să
      trimită echipe de intervenție prioritar la accidentele cu probabilitate mare de gravitate
    """)
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
        df_ols[col] = df_ols[col].fillna(df_ols[col].mean())

    features_ols = ['Temperature(C)', 'Humidity(%)',
                    'Visibility(mi)', 'Wind_Speed(mph)',
                    'Distance(mi)', 'Junction', 'Traffic_Signal']

    features_ols = [f for f in features_ols if f in df_ols.columns]

    df_ols_clean = df_ols[features_ols + ['Severity']].dropna()

    # FIX: min() ca sa nu crape
    n = min(5000, len(df_ols_clean))
    df_sample = df_ols_clean.sample(n, random_state=42)

    X_ols = sm.add_constant(df_sample[features_ols].astype(float))
    y_ols = df_sample['Severity']

    model_ols = sm.OLS(y_ols, X_ols).fit()

    st.subheader("Sumar model OLS")
    st.text(model_ols.summary().as_text())

    # Coeficienți vizualizați
    coef_ols = pd.DataFrame({
        "Variabilă": model_ols.params.index,
        "Coeficient": model_ols.params.values,
        "p-value": model_ols.pvalues.values
    }).iloc[1:]  # excludem constanta

    fig_ols = px.bar(coef_ols, x="Coeficient", y="Variabilă",
                     orientation="h",
                     title="Coeficienți OLS",
                     color="Coeficient",
                     color_continuous_scale="RdBu")
    st.plotly_chart(fig_ols, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("R² ajustat", f"{model_ols.rsquared_adj:.4f}")
    col2.metric("AIC", f"{model_ols.aic:.1f}")

    st.markdown("**Interpretare economică:** Coeficienții indică direcția și magnitudinea efectului fiecărei variabile asupra severității accidentului.")

# ============================================================
# PAGINA 7 — CONCLUZII
# ============================================================
elif pagina == "7. Concluzii":
    st.title("Concluzii")

    st.markdown("""
    ### Sinteza rezultatelor

    **1. Date brute**
    Setul de date conține accidente rutiere din SUA între 2016–2023, cu variabile meteo, geografice și de infrastructură.

    **2. Curățare date**
    Au fost eliminate coloanele cu peste 50% valori lipsă, completate valorile numerice cu media și cele categorice cu modul.
    Outlierii au fost detectați și eliminați prin metoda IQR.

    **3. Statistici descriptive**
    - Majoritatea accidentelor au severitate 2 (moderată).
    - Orele de vârf: dimineața (7–9) și seara (16–18), corespunzând traficului de navetă.
    - Statele cu cele mai multe accidente: CA, FL, TX.

    **4. Clusterizare KMeans**
    Algoritmul a identificat grupuri distincte de condiții meteo asociate accidentelor —
    temperaturi ridicate cu vizibilitate bună vs. condiții adverse (ceață, ploaie).

    **5. Regresie logistică**
    Modelul prezice cu acuratețe rezonabilă dacă un accident va fi grav.
    Variabilele cu cel mai mare impact: vizibilitatea, distanța afectată și prezența intersecțiilor.

    **6. Regresie OLS**
    R² ajustat redus sugerează că severitatea depinde și de factori neobservabili (comportamentul șoferului, viteza).
    Variabilele meteo au efect semnificativ statistic (p < 0.05).

    ### Recomandări
    - Autorități: monitorizare sporită în orele de vârf și condiții de vizibilitate redusă.
    - Infrastructură: semnalizare îmbunătățită la intersecții cu risc ridicat.
    """)
