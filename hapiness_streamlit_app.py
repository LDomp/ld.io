import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns  # Optionnel
import joblib  # Ou utilisez pickle
import plotly.express as px

df = pd.read_csv('Mental.csv')
df['Mental Health Condition'] = df['Mental Health Condition'].fillna('No diagnosis')

# Créer une nouvelle colonne binaire à partir de la colonne "Mental Health Condition"
df['Mental Health Status'] = df['Mental Health Condition'].apply(
    lambda x: 'No Mental Health Condition' if x == 'No diagnosis' else 'Mental Health Condition'
)

# Regrouper les âges
if 'Age Group' not in df.columns:
    age_bins = [0, 25, 35, 45, 60, 100]
    age_labels = ['18-25', '26-35', '36-45', '46-60', '60+']
    df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=True)



# Détection des types de colonnes
categorical_columns = df.select_dtypes(include='object').columns.tolist()
if 'Age Group' not in categorical_columns:
    categorical_columns.append('Age Group')
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

def app_exploration(df):
    st.title("🔍 Exploration des variables")

    st.markdown("Sélectionnez une variable pour explorer sa distribution de manière interactive.")

    # Détection des types de variables
    categorical_columns = df.select_dtypes(include='object').columns.tolist()
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    all_columns = categorical_columns + numerical_columns

    selected_var = st.selectbox("📊 Choisissez une variable :", all_columns)

    if selected_var in categorical_columns:
        st.subheader(f"Distribution de {selected_var} (catégorielle)")

        data = df[selected_var].fillna("Non renseigné")
        counts = data.value_counts().reset_index()
        counts.columns = [selected_var, 'Count']

        chart_type = st.radio("Type de graphique :", ['Camembert', 'Barplot'])

        if chart_type == 'Camembert':
            fig = px.pie(counts, names=selected_var, values='Count',
                         title=f"Répartition de {selected_var}",
                         color_discrete_sequence=px.colors.qualitative.Set3)
        else:
            fig = px.bar(counts, x=selected_var, y='Count',
                         title=f"Répartition de {selected_var}",
                         color=selected_var,
                         color_discrete_sequence=px.colors.qualitative.Set3)

        st.plotly_chart(fig)

    else:  # numérique
        st.subheader(f"Distribution de {selected_var} (numérique)")

        chart_type = st.radio("Type de graphique :", ['Histogramme', 'Boxplot'])

        if chart_type == 'Histogramme':
            fig = px.histogram(df, x=selected_var,
                               nbins=30,
                               title=f"Distribution de {selected_var}",
                               color_discrete_sequence=['#00CC96'])
        else:
            fig = px.box(df, y=selected_var,
                         title=f"Répartition de {selected_var}",
                         color_discrete_sequence=['#AB63FA'])

        st.plotly_chart(fig)


def app_relations(df):

    st.title("🔗 Relations entre variables")

    st.markdown("Explorez les relations entre deux variables numériques, avec des filtres interactifs.")

    # Variables numériques et catégorielles
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include='object').columns.tolist()

    # Sélection des variables X et Y
    x_var = st.selectbox("📈 Variable X :", numerical_columns, index=0)
    y_var = st.selectbox("📉 Variable Y :", numerical_columns, index=1)

    # Choix de la variable de filtre (catégorielle)
    filter_col = st.selectbox("🎯 Filtrer par :", ["Aucun filtre"] + categorical_columns)

    if filter_col != "Aucun filtre":
        unique_vals = df[filter_col].dropna().unique().tolist()
        selected_vals = st.multiselect(f"Sélectionnez les valeurs de '{filter_col}' :", unique_vals, default=unique_vals)

        # Appliquer le filtre
        filtered_df = df[df[filter_col].isin(selected_vals)]
    else:
        filtered_df = df.copy()

    # Graphique interactif avec régression
    fig = px.scatter(
        filtered_df,
        x=x_var,
        y=y_var,
        color=filter_col if filter_col != "Aucun filtre" else None,
        trendline="ols",
        title=f"Relation entre {x_var} et {y_var}",
        opacity=0.7
    )

    st.plotly_chart(fig)

    
def app_comparaisons(df):
    import streamlit as st
    import plotly.express as px

    st.title(" Comparaison par groupes")

    st.markdown("Visualisez comment une variable numérique varie selon les groupes d'une variable catégorielle.")



    # Sélection des variables
    y_var = st.selectbox("🎯 Variable numérique à comparer :", numerical_columns)
    group_col = st.selectbox("🧩 Grouper par :", categorical_columns)

    # Type de graphique
    plot_type = st.radio("Type de graphique :", ['Boxplot', 'Violin'])

    # Nettoyage des NaN
    df_filtered = df[[group_col, y_var]].dropna()

    # Graphique
    if plot_type == 'Boxplot':
        fig = px.box(df_filtered, x=group_col, y=y_var, color=group_col,
                     title=f"{y_var} par {group_col}",
                     color_discrete_sequence=px.colors.qualitative.Set2)
    else:
        fig = px.violin(df_filtered, x=group_col, y=y_var, color=group_col, box=True,
                        title=f"{y_var} par {group_col}",
                        color_discrete_sequence=px.colors.qualitative.Set2)

    st.plotly_chart(fig)
 
def app_heatmap(df):

    st.title("Analyse des corrélations")
    st.markdown("Visualisez les corrélations entre les variables numériques de manière personnalisée.")

    # Sélection des variables numériques
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    selected_vars = st.multiselect(
        "Sélectionnez les variables à inclure dans la heatmap :",
        options=numeric_cols,
        default=numeric_cols  # Toutes sélectionnées par défaut
    )

    if len(selected_vars) < 2:
        st.warning("Veuillez sélectionner au moins deux variables.")
        return

    # Calcul des corrélations
    corr_matrix = df[selected_vars].corr()

    # Création de la heatmap avec style sombre
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='black')

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=0.5,
        linecolor='gray',
        cbar=True,
        square=True,
        ax=ax
    )

    ax.set_title("Corrélations entre variables numériques", fontsize=14, color='white')
    plt.xticks(rotation=45, ha='right', color='white')
    plt.yticks(color='white')
    plt.tight_layout()

    st.pyplot(fig)
    
st.title("🧠 Modes de vie et bonheur")
st.sidebar.title("Sommaire")
pages=["Introduction", "Exploration des données", "Relations entre variables", "Insights clés et interprétations", "Conclusions"]
page=st.sidebar.radio("Sélectionnez une partie", pages)

st.sidebar.markdown("[Luc DOMPEYRE](https://www.linkedin.com/in/luc-dompeyre/)")

if page == pages[0] :
    st.title("Introduction")
    st.markdown("""
    La santé mentale est un sujet de plus en plus important dans notre société moderne. 
    Elle englobe notre bien-être émotionnel, psychologique et social, et influence notre façon de penser, d'agir et de ressentir.
    """)
    st.markdown("""
    Ce projet explore les liens entre les habitudes de vie (sommeil, alimentation, sport, stress…) et le bonheur, à travers un dataset de 3 000 individus collecté entre 2019 et 2024.
    Nous allons examiner les relations entre différentes variables, visualiser les données et comprendre quels sont les habitudes qui permettent d'être plus heureux.
    """)
    st.image("Live-longer.webp" , caption="Mode de vie et bonheur", use_column_width=True)
    
if page == pages[1]:
    st.markdown("""
    Dans cette section, nous allons explorer les données de manière interactive. 
    Vous pouvez choisir différentes variables pour visualiser leur distribution et leurs relations.
    """)
    
    app_exploration(df)


    st.markdown("""
    ### 🧍‍♂️🧍‍♀️ Qui sont les répondants ?

    #### Âge :
    - **Moyenne** : 41,2 ans  
    - **Médiane** : 41 ans  
    - **Plage d’âge** : 18 à 64 ans  
    - 50% des répondants ont entre **30 et 53 ans**

    >  Une population adulte active, bien répartie sur les tranches d’âge courantes.

    #### Genre :
    - **Female** : 34,1%
    - **Other** : 33,2%
    - **Male** : 32,7%

    > Répartition relativement équilibrée, avec une légère surreprésentation de la catégorie *Other* (probablement une désignation inclusive).

    #### 🌍 Pays principaux :
    - **USA** : 14,9%
    - **Japon** : 14,6%
    - **Australie** : 14,5%
    - **Inde** : 14,5%
    - **Canada** : 14,3%
    - **Brésil** : 13,8%
    - **Allemagne** : 13,5%

    > Les répondants sont répartis équitablement entre plusieurs pays, offrant une bonne diversité.

    ---

    ### 😊 Distribution des scores de bonheur

    - **Moyenne** : 5,4  
    - **Médiane** : 5,4  
    - **Min/Max** : de 1 à 10  
    - **Écart-type** : 2,56  
    - 50% des répondants ont un score entre **3,2 et 7,5**

    > La distribution semble étalée et centrée. Un histogramme permettrait de confirmer une éventuelle bimodalité ou dispersion.

    ---

    ### ⚖️ Déséquilibres dans les données

    - **Santé mentale** : 80% des répondants ont un diagnostic de santé mentale, le jeu de données peut être très efficace pour analyser les liens entre santé mentale et les habitudes de vie.
    - **Pays** : 7 pays dominants → les autres sont sous-représentés
    - **Âge** : distribution régulière mais majorité entre 30–53 ans

    > Globalement, le dataset est équilibré et fiable pour l’analyse
    """)

if page == pages[2]:
    st.markdown("""
    Dans cette section, nous allons examiner les relations entre deux variables numériques. 
    Vous pouvez filtrer les données en fonction de variables catégorielles pour explorer des sous-ensembles spécifiques.
    """)
    
    app_relations(df)
    app_comparaisons(df)
    app_heatmap(df)
    
if page == pages[3]:
    st.markdown("""
    Pour rappel, ce projet de data analyse vise à **comprendre les liens entre le mode de vie et le bonheur subjectif** à travers un jeu de données contenant plus de 3 000 observations collectées entre 2019 et 2024.

    Nous explorons ici l’impact de plusieurs facteurs comme :
    - le **stress**,
    - l’**activité physique**,
    - l’**alimentation**,
    - la **santé mentale**,
    - ou encore le **genre**…

    … sur le **niveau de bonheur auto-déclaré** des individus.
    
    ---
    """)
    
    st.markdown("""
    ### Niveau de stress vs Niveau de bonheur

    | Stress Level | Moyenne | Médiane | Écart-type |
    |--------------|---------|---------|------------|
    | High         | 5.44    | 5.4     | 2.55       |
    | Low          | 5.41    | 5.5     | 2.61       |
    | Moderate     | 5.34    | 5.4     | 2.52       |


    ### Analyse :

    - Les différences de moyennes sont faibles, mais on note que :
    - Les personnes avec un stress **modéré** ont en moyenne un **score de bonheur légèrement plus bas**.
    - Les personnes **très stressées ou peu stressées** présentent des scores similaires.

    - L’écart-type élevé (~2.5) dans chaque groupe montre une **forte variabilité**.

    - La médiane est quasiment identique pour tous les groupes.

    > ➡️ **Conclusion** : Le niveau de stress n’a pas d’impact direct fort sur le bonheur dans ce dataset. Pour autant, il ne faut pas négliger le stress sur les autres facteurs et notamment la santé mentale qui a elle une importance directe plus élevée sur le niveau de bonheur.
    ---
    
    """)
    
    st.markdown("""
    ### Niveau d’exercice vs Niveau de bonheur

    | Exercise Level | Moyenne | Médiane | Écart-type |
    |----------------|---------|---------|------------|
    | High           | 5.55    | 5.7     | 2.61       |
    | Moderate       | 5.36    | 5.4     | 2.54       |
    | Low            | 5.29    | 5.2     | 2.51       |



    ### Analyse :

    - Plus le niveau d’exercice est élevé, plus le **score moyen de bonheur augmente**.
    - La **différence entre “High” et “Low” est de 0.26 points**, ce qui est **plus marqué** que pour la variable stress.
    - La médiane et le 3e quartile sont également plus élevés chez les plus actifs.

    > ➡️ **Conclusion** : Le niveau d’activité physique semble être un **levier positif mesurable sur le bonheur**. Une activité régulière pourrait jouer un rôle important dans l'amélioration du bien-être mental.
    ---
    
    """)
    
    st.markdown("""
    ### Régime alimentaire vs Niveau de bonheur

    | Diet Type         | Moyenne | Médiane | Écart-type |
    |-------------------|---------|---------|------------|
    | Balanced          | 5.58    | 5.7     | 2.59       |
    | High-Protein      | 5.35    | 5.4     | 2.50       |
    | Low-Carb          | 5.19    | 5.2     | 2.50       |
    | Mediterranean     | 5.39    | 5.4     | 2.54       |
    | Processed         | 5.08    | 5.0     | 2.50       |
    | Vegan             | 5.34    | 5.5     | 2.59       |



    ### Analyse :

    - Le régime **le plus associé au bonheur est le “Balanced”**, suivi du “Mediterranean”.
    - Le régime **“Processed” affiche le score moyen de bonheur le plus bas (5.08)**.
    - Les personnes ayant une alimentation équilibrée présentent également une médiane plus haute.

    > ➡️ **Conclusion** : Une alimentation équilibrée semble **favoriser un meilleur bien-être subjectif**, tandis que les régimes riches en aliments transformés sont associés à des niveaux de bonheur plus faibles.
    ---
    """)
    
    st.markdown("""
    ### 🧠 Trouble mental vs Niveau de bonheur

    | Condition mentale | Moyenne | Médiane | Écart-type |
    |-------------------|---------|---------|------------|
    | Anxiety           | 5.26    | 5.15    | 2.56       |
    | Bipolar           | 5.47    | 5.40    | 2.53       |
    | Depression        | 5.34    | 5.50    | 2.54       |
    | None              | 5.45    | 5.50    | 2.55       |
    | PTSD              | 5.46    | 5.60    | 2.61       |


    ### Analyse :

    - Les personnes sans trouble déclaré (`None`) présentent un score de bonheur **légèrement supérieur à la moyenne** (5.45), mais pas de façon spectaculaire.
    - Les scores sont **relativement homogènes**, tous entre **5.2 et 5.5** de moyenne.
    - Les personnes avec **Anxiety** ont le score de bonheur **le plus faible** du groupe (5.26).
    - **PTSD** et **Bipolar** affichent des moyennes proches ou supérieures à ceux sans trouble, ce qui peut surprendre.

    > ➡️ **Conclusion** :
    > Contrairement à ce qu’on pourrait penser, la présence d’un trouble mental **n’entraîne pas systématiquement un score de bonheur très bas**.
    > Cela peut refléter :
    > - un **accompagnement ou traitement**,
    > - une **variabilité interindividuelle importante**,
    > - ou une **résilience personnelle**.

    > Ces résultats confirment la complexité de la santé mentale et son lien au bien-être : **un diagnostic ne résume pas une personne**, ni son bonheur.
    """)
    
    st.markdown("""
    ### Genre vs Niveau de bonheur

    | Genre   | Moyenne | Médiane | Écart-type |
    |---------|---------|---------|------------|
    | Female  | 5.33    | 5.3     | 2.55       |
    | Male    | 5.46    | 5.5     | 2.58       |
    | Other   | 5.41    | 5.5     | 2.58       |

    ### Analyse :

    - Les scores de bonheur sont **très proches entre les trois genres**.
    - Les hommes ont une **moyenne légèrement plus élevée** (5.46), suivis de près par les personnes se déclarant comme “Other” (5.41).
    - Les femmes sont très légèrement en dessous, mais la différence est **très faible (< 0.15 pt)**.

    > ➡️ **Conclusion** : Nous nous y attendions mais il n’y a **pas de disparité significative de bonheur selon le genre** dans ce dataset. Cela suggère une perception du bien-être relativement stable entre les identités de genre représentées.
    ---
    
    """)
    
    st.markdown("## 🔎 Analyse des autres facteurs influençant le bonheur")

    st.markdown("""
    Nous avons examiné plusieurs variables numériques susceptibles d’avoir un lien avec le **Happiness Score** :

    | Variable                         |  Corrélation avec le bonheur |
    |----------------------------------|------------------------------|
    | 💤 Sleep Hours                   | **+0.017**                   |
    | 📱 Screen Time per Day (Hours)   | **+0.017**                   |
    | 🧑‍💼 Work Hours per Week           | **+0.011**                   |
    | 🎂 Age                           | **–0.016**                   |
    | 🫂 Social Interaction Score      | **–0.040**                   |

   

    ### Interprétation :

    - Toutes les **corrélations sont très faibles**, proches de zéro.
    - Aucun de ces facteurs n’a de **relation linéaire significative** avec le bonheur.
    - Même les interactions sociales, qui pourraient être vues comme positives, montrent une **corrélation négative faible**.


    ### ➡️ Conclusion :

    > Ces résultats suggèrent que :
    > - Leur impact est **non linéaire** ou **conditionnel** (ex : dépend du stress ou de la santé mentale).
    > - Leur influence est **complexe** et peut varier d’un individu à l’autre.
    > - Des **analyses croisées** ou par **tranches** seraient nécessaires pour mieux les comprendre.

    ---

    """)

    
if page == pages[4]:
    st.markdown("""
    # Synthèse

    Après analyse des différentes variables de mode de vie, voici les **principaux enseignements** :

    - 🏃‍♂️ **L’exercice physique** a un **impact clair et positif** sur le bonheur : plus on bouge, plus on est heureux.
    - 🥗 **L’alimentation équilibrée** est également associée à des scores de bonheur plus élevés que les régimes transformés.
    - 🧠 **La santé mentale** est le facteur le plus fortement lié au bonheur : une absence de troubles mentaux augmente le score moyen de +1.1 point.
    - 😮‍💨 Le **stress**, quant à lui, n’a pas montré d’influence marquée, peut-être en raison de sa complexité ou d'effets indirects.
    - 🚻 **Le genre** n’influence pas significativement le niveau de bonheur dans ce jeu de données.
    - 📊 Les autres facteurs (sommeil, temps d’écran, heures de travail) n’ont pas montré de corrélations significatives avec le bonheur mais ces facteurs restent à creuser.
    
    **Conclusion générale** :
    > Bien que cette analyse soit incomplète, elle met en lumière des **tendances intéressantes**.
    > Le mode de vie **influence bien le bonheur**, notamment via l’activité physique, l’alimentation et la santé mentale.  
    > Ces résultats confirment que **préserver son équilibre personnel et corporel** est une voie concrète vers un mieux-être.

    **Et après ?**  
    Ce dataset offre **un immense potentiel d’analyse** :  
    - Prédiction du bonheur par machine learning  
    - Détection de profils à risque  
    - Études d’impact multi-variées  
    - Exploration géographique et culturelle  
    - Analyse temporelle (pré/post-COVID)  

    > Ce projet est une première pierre. D’autres explorations peuvent suivre dans un cadre RH, médical, sociétal… ou personnel.
    """)