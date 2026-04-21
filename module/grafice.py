import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def grafic_severitate_bar(sev_df):
    fig = px.bar(sev_df, x='Severitate', y='Nr. accidente',
                 title='Nr. accidente pe nivel de severitate',
                 color='Nr. accidente',
                 color_continuous_scale='Reds',
                 text='Nr. accidente')
    fig.update_traces(texttemplate='%{text:,}', textposition='outside')
    return fig


def grafic_severitate_pie(sev_df):
    return px.pie(sev_df, names='Severitate', values='Nr. accidente',
                  title='Distribuția procentuală a severității',
                  color_discrete_sequence=px.colors.sequential.Reds_r)


def grafic_accidente_ora(ora_df):
    fig = px.line(ora_df, x='Ora', y='nr_accidente',
                  title='Numărul de accidente per oră',
                  markers=True,
                  labels={"Ora": "Ora din zi", "nr_accidente": "Nr. accidente"})
    fig.add_vrect(x0=6.5, x1=9.5, fillcolor="orange",
                  opacity=0.15, annotation_text="Vârf dimineață")
    fig.add_vrect(x0=15.5, x1=18.5, fillcolor="red",
                  opacity=0.15, annotation_text="Vârf seară")
    return fig


def grafic_accidente_an(an_df):
    fig = px.bar(an_df, x='Year', y='nr_accidente',
                 title='Numărul de accidente per an',
                 color='nr_accidente',
                 color_continuous_scale='Blues',
                 text='nr_accidente')
    fig.update_traces(texttemplate='%{text:,}', textposition='outside')
    return fig


def grafic_top_state(state_df):
    fig = px.bar(state_df, x='State', y='nr_accidente',
                 title='Top 15 state după numărul de accidente',
                 color='severitate_medie',
                 color_continuous_scale='RdYlGn_r',
                 text='nr_accidente',
                 labels={"nr_accidente": "Nr. accidente",
                         "severitate_medie": "Severitate medie"})
    fig.update_traces(texttemplate='%{text:,}', textposition='outside')
    return fig


def grafic_severitate_meteo(meteo_df):
    fig = px.bar(meteo_df, x='Weather_Condition', y='severitate_medie',
                 title='Severitate medie pe condiții meteo',
                 color='severitate_medie',
                 color_continuous_scale='Oranges',
                 text='severitate_medie',
                 labels={"Weather_Condition": "Condiție meteo",
                         "severitate_medie": "Severitate medie"})
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def grafic_zi_noapte_bar(zn_df, y_col, title):
    fig = px.bar(zn_df, x='Sunrise_Sunset', y=y_col,
                 title=title,
                 color='Sunrise_Sunset',
                 text=y_col,
                 labels={"Sunrise_Sunset": "Moment zilei"})
    fig.update_traces(textposition='outside')
    return fig


def grafic_outlieri(outlieri_info):
    fig = px.bar(
        x=list(outlieri_info.keys()),
        y=list(outlieri_info.values()),
        labels={"x": "Coloană", "y": "Nr. outlieri eliminați"},
        title="Numărul de outlieri eliminați per variabilă",
        color=list(outlieri_info.values()),
        color_continuous_scale="Reds",
        text=list(outlieri_info.values())
    )
    fig.update_traces(texttemplate='%{text:,}', textposition='outside')
    return fig


def grafic_valori_lipsa(df):
    lipsa = df.isnull().sum().reset_index()
    lipsa.columns = ["Coloană", "Valori lipsă"]
    lipsa["Procent (%)"] = (lipsa["Valori lipsă"] / len(df) * 100).round(2)
    lipsa = lipsa[lipsa["Valori lipsă"] > 0].sort_values("Valori lipsă", ascending=False)
    fig = px.bar(lipsa, x="Coloană", y="Procent (%)",
                 title="Procentul valorilor lipsă per coloană",
                 color="Procent (%)",
                 color_continuous_scale="Reds",
                 text="Procent (%)")
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    return fig, lipsa


def grafic_elbow(K_range, inertii):
    fig = px.line(x=K_range, y=inertii,
                  labels={"x": "K (număr clustere)", "y": "Inerție"},
                  title="Metoda cotului — identificarea K optim",
                  markers=True)
    fig.update_traces(line_color='#E24B4A', marker_size=8)
    return fig


def grafic_scatter_clustere(df_cl, x_col, y_col, title, x_label, y_label):
    return px.scatter(df_cl, x=x_col, y=y_col,
                      color='Cluster',
                      title=title,
                      labels={x_col: x_label, y_col: y_label},
                      opacity=0.5)


def grafic_distributie_clustere_bar(dist_df):
    fig = px.bar(dist_df, x='Cluster', y='Nr. accidente',
                 title='Numărul de accidente per cluster',
                 color='Cluster', text='Nr. accidente')
    fig.update_traces(texttemplate='%{text:,}', textposition='outside')
    return fig


def grafic_distributie_clustere_pie(dist_df):
    return px.pie(dist_df, names='Cluster', values='Nr. accidente',
                  title='Distribuția procentuală pe clustere')


def grafic_severitate_cluster(sev_cl):
    fig = px.bar(sev_cl, x='Cluster', y='Severitate medie',
                 title='Severitatea medie a accidentelor per cluster',
                 color='Severitate medie',
                 color_continuous_scale='Reds',
                 text='Severitate medie')
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(yaxis_range=[1, 4])
    return fig


def grafic_confusion_matrix(cm):
    return px.imshow(cm, text_auto=True,
                     labels=dict(x="Predicție", y="Real"),
                     x=['Negrav', 'Grav'], y=['Negrav', 'Grav'],
                     title="Matricea de confuzie",
                     color_continuous_scale="Blues")


def grafic_coeficienti_logistic(features, coef):
    coef_df = pd.DataFrame({
        "Variabilă": features,
        "Coeficient": coef
    }).sort_values("Coeficient")
    fig = px.bar(coef_df, x="Coeficient", y="Variabilă",
                 orientation="h",
                 title="Importanța variabilelor — coeficienți regresie logistică",
                 color="Coeficient",
                 color_continuous_scale="RdBu")
    return fig


def grafic_coeficienti_ols(coef_df):
    coef_plot = coef_df[coef_df['Variabilă'] != 'const'].copy()
    fig = px.bar(coef_plot, x='Coeficient', y='Variabilă',
                 orientation='h',
                 title='Coeficienții modelului OLS',
                 color='Coeficient',
                 color_continuous_scale='RdBu',
                 text='Coeficient')
    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig.add_vline(x=0, line_dash='dash', line_color='gray')
    return fig


def grafic_reziduale(fitted, residuals, n=3000):
    fig = px.scatter(x=fitted[:n], y=residuals[:n],
                     labels={"x": "Valori estimate (ŷ)", "y": "Reziduale (y - ŷ)"},
                     title="Reziduale vs. Valori estimate",
                     opacity=0.3)
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    return fig


def grafic_distributie_reziduale(residuals):
    return px.histogram(x=residuals, nbins=50,
                        title="Distribuția reziduelor",
                        labels={"x": "Reziduală", "y": "Frecvență"},
                        color_discrete_sequence=["#378ADD"])


def grafic_real_vs_estimat(y_real, y_estimat, n=50):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(n)),
                              y=y_real[:n],
                              mode='markers', name='Real',
                              marker=dict(color='#E24B4A', size=6)))
    fig.add_trace(go.Scatter(x=list(range(n)),
                              y=y_estimat[:n].round(2),
                              mode='lines', name='Estimat',
                              line=dict(color='#378ADD', width=2)))
    fig.update_layout(title='Severitate reală vs. estimată',
                      xaxis_title='Observație',
                      yaxis_title='Severitate')
    return fig
