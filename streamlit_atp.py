import streamlit as st
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import io

df = pd.read_csv('atp_data.csv')

st.title("Paris Sportifs Tennis 🎾")
st.sidebar.title("Sommaire")
pages=["Introduction", "Exploration", "Visualisation", "Nettoyage des données", "Préparation pour l'entrainement des modèles","Machine learning", "Calcul des probabilités", "Conclusion"]
page=st.sidebar.radio("Sélectionnez une partie", pages)


st.sidebar.markdown("[Luc DOMPEYRE](https://www.linkedin.com/in/luc-dompeyre/)")



if page == pages[0] :
    st.write("### Introduction")
  
    st.write("Dans ce projet, mon objectif est de **prédire de manière plus précises que celles des bookmakers les résultats des matchs de tennis ATP** et ceci via l'entraînement d'un modèle performant de Machine Learning.")
  
    st.write("Avant d’attaquer la modélisation, j'ai exploré et nettoyé les données afin d’obtenir un dataset fiable et exploitable. J'ai également identifié et corrigé des erreurs d'exécution pour assurer la cohérence des analyses.")
    st.image("pexels-rajtatavarthy-171568.jpg")
  
if page == pages[1] :
    st.write("### Exploration")
  
    st.write("J'ai d'abord examiné la structure du dataset en vérifiant les types de données, la présence de valeurs manquantes et d’éventuelles incohérences. Cette première analyse m'a permis d’anticiper et d’éviter des erreurs d’exécution lors du traitement des données.")
  
    st.write("Les données utilisées dans cette étude proviennent de [Kaggle](https://www.kaggle.com/datasets/edouardthomas/atp-matches-dataset) et comprennent 44709 lignes et 23 colonnes. Elles regroupent diverses informations sur les matchs ATP, comme les joueurs, les surfaces, les classements et les côtes des bookmakers. Une exploration approfondie a été nécessaire pour comprendre la pertinence de chaque variable et assurer une utilisation optimale dans mes modèles prédictifs.")
  
    st.write("### Présentation des variables du dataset :")
    st.write("<span style='color:red'>Location</span> : Localisation du tournoi", unsafe_allow_html=True)
    st.write("<span style='color:red'>Tournament</span> : Nom du tournoi", unsafe_allow_html=True)
    st.write("<span style='color:red'>Date</span> : Date du match", unsafe_allow_html=True)
    st.write("<span style='color:red'>Series, Court, Surface</span> : Type de compétition et surface du match", unsafe_allow_html=True)
    st.write("<span style='color:red'>Round, Best of</span> : Tour du tournoi et format du match (meilleur des 3 ou 5 sets)", unsafe_allow_html=True)
    st.write("<span style='color:red'>Winner, Loser</span> : Noms du vainqueur et du perdant", unsafe_allow_html=True)
    st.write("<span style='color:red'>WRank, LRank</span> : Classement ATP du vainqueur et du perdant", unsafe_allow_html=True)
    st.write("<span style='color:red'>Wsets, Lsets</span> : Nombre de sets gagnés par le vainqueur et par le perdant", unsafe_allow_html=True)
    st.write("<span style='color:red'>Comment</span> : Indique si le match a été terminé normalement ou si un événement est venu perturbé l’issu du match", unsafe_allow_html=True) 
    st.write("<span style='color:red'>PSW, PSL, B365W, B365L</span> : Côtes des bookmakers Pinnacles Sport et Bet365 pour les joueurs", unsafe_allow_html=True)
    st.write("<span style='color:red'>elo_winner, elo_loser</span> : Classement Elo des joueurs", unsafe_allow_html=True)
    st.write("<span style='color:red'>proba_elo</span> : Probabilité de victoire qu’avait le vainqueur basée sur le classement Elo", unsafe_allow_html=True)

  
    st.write("### Affichage des premières lignes du dataset pour avoir un premier aperçu:")
    st.dataframe(df.head(10))
  
    st.header("Informations sur le jeu de données")

# Capture la sortie de df.info() dans un buffer
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_string = buffer.getvalue()

    # Affiche les informations dans un conteneur code
    st.code(info_string, language="text")
    
    st.write("Après vérification des types des variables, je remarque que la variable <span style='color:red'>Date</span> n’est pas au format **‘datetime’**, je modifierai son type dans la préparation du jeu de données.", unsafe_allow_html=True)
  
    st.write('Le **nombre de doublons** est :',df.duplicated().sum())
  
    st.write("### Nombre de valeurs manquantes par lignes")
    st.write(df.isnull().sum())

if page == pages[2] :
    st.title("Visualisation")

#1er graphique
  
  # Création du pie chart avec pourcentage
    surface_distribution = df['Surface'].value_counts()
    st.write("### Distribution des matchs par type de surface")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
    surface_distribution,
    labels=surface_distribution.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=sns.color_palette("Set2", len(surface_distribution)))
    st.pyplot(fig)

#2eme graphique
    def target(row):
            return row["WRank"] <= row["LRank"]

    df['target'] = df.apply(lambda row: target(row), axis=1)

# Interface Streamlit
    st.write("### Comparaison du classement du vainqueur vs classement du perdant")
    st.write(" ")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
    x='WRank', y='LRank', data=df, hue='target',
    palette={True: 'blue', False: 'green'}, s=50, alpha=0.6, ax=ax
    )

    # Tracer une ligne diagonale (y = x)
    ax.plot([df['WRank'].min(), df['WRank'].max()],
    [df['WRank'].min(), df['WRank'].max()],
    color='red', linestyle='dashed', linewidth=2, label='Égalité de classement')

    # Légendes et titres
    ax.legend(title='Gagnant mieux classé', loc='upper right')
    ax.set_title('Nuage de points entre le classement du gagnant et du perdant', fontsize=14)
    ax.set_xlabel('Classement du gagnant (WRank)', fontsize=12)
    ax.set_ylabel('Classement du perdant (LRank)', fontsize=12)

    # Affichage du graphique dans Streamlit
    st.pyplot(fig)
    st.write(" ")
    
#3eme graphique
# Calcul de la distribution en pourcentage
    comment_distribution = df['Comment'].value_counts(normalize=True) * 100

# Interface Streamlit
    st.write("### Distribution des matchs par type de commentaire")
    st.write(" ")
    # Création du graphique
    fig, ax = plt.subplots(figsize=(10, 6))
    comment_distribution.plot(kind='bar', color=sns.color_palette("Set3", len(comment_distribution)), ax=ax)

    # Ajouter un titre et des labels
    ax.set_xlabel('Type de commentaire', fontsize=12)
    ax.set_ylabel('Pourcentage de matchs (%)', fontsize=12)

    # Ajouter les valeurs sur chaque barre
    for i, v in enumerate(comment_distribution):
        ax.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10)

    # Afficher le graphique dans Streamlit
    st.pyplot(fig)
    st.write(" ")

#4eme graphique
# Calcul des pourcentages de valeurs manquantes
    missing_data = df.isnull().mean() * 100
    missing_data = missing_data.sort_values(ascending=False)

# Interface Streamlit
    st.write("### Analyse des valeurs manquantes")
    st.write(" ")

    # Création du graphique
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=missing_data.index, y=missing_data.values, palette='viridis', ax=ax)

    # Ajouter des titres et labels
    ax.set_title('Pourcentage de valeurs manquantes par colonne', fontsize=14)
    ax.set_xlabel('Colonnes', fontsize=12)
    ax.set_ylabel('Pourcentage de valeurs manquantes (%)', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)  # Rotation pour lisibilité

    # Affichage dans Streamlit
    st.pyplot(fig)
    st.write(" ")

#5eme graphique
# Interface Streamlit
    st.write("### Détection des valeurs aberrantes avec Boxplots")
    st.write(" ")
    # Création du graphique avec 6 subplots (2 lignes, 3 colonnes)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    groupes = {
        'ATP': ['ATP'],
        'Rankings': ['WRank', 'LRank'],
        'Sets': ['Best of', 'Lsets', 'Wsets'],
        'Cotes': ['PSW', 'PSL', 'B365W', 'B365L'],
        'Elo': ['elo_winner', 'elo_loser'],
        'Proba_elo': ['proba_elo']}

    # Boucle pour créer les boxplots dans chaque sous-figure
    for i, (groupe, variables) in enumerate(groupes.items()):    
        ax = axes[i // 3, i % 3]  # Position dans la grille
        sns.boxplot(data=df[variables], ax=ax, palette="Set2")  # Création du boxplot
        ax.set_title(f'Boxplot pour {groupe}', fontsize=14)
        ax.set_xlabel('Variables', fontsize=12)
        ax.set_ylabel('Valeurs', fontsize=12)
        ax.tick_params(axis='x', rotation=45)

    # Ajustement du layout pour éviter le chevauchement des titres
    plt.tight_layout()

    # Affichage du graphique dans Streamlit
    st.pyplot(fig)
    st.write(" ")

#6eme graphique
    st.write("### Afficher la distribution de la variable cible")
    st.write(" ")

    def target(row):
        if row["WRank"] > row["LRank"]:
            return 1
        else:
            return 0
    df['target'] = df.apply(lambda row: target(row), axis = 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=df['target'].value_counts().index,
    y=df['target'].value_counts().values,
    palette='viridis')

# Ajouter les annotations directement sur les barres
    values = df['target'].value_counts().values
    labels = df['target'].value_counts().index

    for i in range(len(values)):
        plt.text(i, values[i] + 0.5, str(values[i]), ha='center', fontsize=12, fontweight='bold')

    ax.set_title("Répartition de la variable cible", fontsize=14)
    ax.set_xlabel("Classes", fontsize=12)
    ax.set_ylabel("Nombre d'occurrences", fontsize=12)
    st.pyplot(fig)


df = df.drop(['B365W', 'B365L', 'PSL', 'PSW', 'Winner', 'Loser', 'Tournament', 'Location'], axis = 1)

df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month

df = df[df["Comment"] == "Completed"]

df = df.drop(['Comment'], axis = 1)
df = df.drop(['Date'], axis = 1)

def target(row):
    if row["WRank"] > row["LRank"]:
        return 1
    else:
        return 0
df['target'] = df.apply(lambda row: target(row), axis = 1)
df = df.drop(columns=['Wsets', 'Lsets', 'proba_elo'])

if page == pages[3] :
    st.title("Nettoyage des données")
  
    st.write("Après cette première exploration, l'objectif est de nettoyer le dataset pour éliminer les erreurs, incohérences et valeurs manquantes qui pourraient fausser les analyses, réduire la précision des modèles et conduire à des décisions erronées. Voici les modifications réalisées :")
  
    st.write("### - Modification du format de la date")
    st.write("Comme expliqué précédemment, la variable <span style='color:red'>Date</span> n’était pas dans un format exploitable (conversion au format ‘datetime’). De plus, je souhaite identifier les tendances saisonnières dans les performances des joueurs ou des tournois donc je créé une nouvelle variable numérique <span style='color:red'>Month</span> qui représente le mois du match (format MM) à partir de la variable <span style='color:red'>Date</span>.", unsafe_allow_html=True)
  
    st.write("### - Suppression des colonnes inutiles")
    st.write("Je décide ensuite de supprimer les variables <span style='color:red'>PSW, PSL, B365W, B365L</span>. Elles correspondent aux côtes des bookmakers qui seront utilisées à des fins ultérieures pour comparer les résultats mon modèle. De plus, les variables <span style='color:red'>Winner et Loser</span> qui font référence aux noms des vainqueurs et des perdants de chaque match ont aussi été supprimées puisqu’elles n’apportent aucune information pour prédire le vainqueur. A l’inverse, la variable <span style='color:red'>proba_elo</span> qui correspond à la probabilité qu’avait le vainqueur de gagner d’après son classement Elo, donne trop d’information quant à l’identité du vainqueur au modèle donc je la supprime également.", unsafe_allow_html=True)
    st.write("Puis, je remarque que les colonnes <span style='color:red'>Location et Tournament</span> sont assez similaires de par la fréquence de leurs modalités. Si une variable a une distribution de ses valeurs relativement égale (c'est-à-dire uniforme), cela signifie qu'elle ne présente pas de tendance ou de structure exploitable par un modèle de machine learning. En d'autres termes, elle ne varie pas de manière corrélée avec la variable cible à prédire. C’est pourquoi, je décide de les supprimer à leur tour.", unsafe_allow_html=True)
    
    st.write("### - Gestion des valeurs manquantes")
    st.write("En examinant les lignes du jeu de données, je remarque que les valeurs manquantes se trouvent presque uniquement dans les matchs qui n’ont pas été terminés (modalité de <span style='color:red'>Comment</span> autre que **‘Completed’**). Pour nettoyer les données, on décide de conserver uniquement les matchs terminés. A la suite de cette opération **il reste une unique ligne** où les valeurs <span style='color:red'>Wsets et Lsets</span> ne sont pas renseignées. ", unsafe_allow_html=True)
        
    st.write("### - Suppression de nouvelles colonnes")
    st.write("Après avoir créé une nouvelle variable <span style='color:red'>Month</span>, la variable <span style='color:red'>Date</span> n’est plus utile. De plus, après avoir supprimé toutes les modalités de la variable <span style='color:red'>Comment</span> pour ne laisser que **‘Completed’**, celle-ci devient également inutile mes modèle. Je les supprime toutes les deux.", unsafe_allow_html=True) 
    st.write("De plus, les variables <span style='color:red'>Wsets et Lsets</span> correspondent respectivement au nombre de sets gagnés par le vainqueur et au nombre de sets gagnés par le perdant, ce sont des informations connues seulement après un match donc qui ne doivent pas être intégrées au modèle par souci de fuite des données. L’objectif du modèle est de prédire l’issue d’un match avant qu’il n’ait eu lieu. Suite à ces suppressions, **il ne reste plus aucune valeur manquante** dans les données.", unsafe_allow_html=True)
        
    st.write("### - Création d'un variable cible")
    st.write("Dans un modèle de Machine Learning, il est essentiel d’avoir une variable cible qui représente la valeur que j'essaye de prédire : “quel joueur va gagner le match ?”. Le modèle ajuste ensuite ses poids et coefficients en fonction des erreurs qu’il commet par rapport à la variable cible.")
    st.write("Je créé ainsi la variable <span style='color:red'>target</span> qui servira de variable cible. Le sujet concerne un problème de classification binaire : le vainqueur est l’un des 2 joueurs. C’est pourquoi j'utilise une condition sur les variables <span style='color:red'>WRank</span> (classement du gagnant) et <span style='color:red'>LRank</span> (classement du perdant) pour la créer.", unsafe_allow_html=True)
    st.write("Celle-ci prend les valeurs suivantes :")
    st.write("0 : Le joueur le mieux classé au classement ATP gagne le match (victoire attendue).")
    st.write("1 : Le joueur le moins bien classé au classement ATP gagne le match (victoire surprenante).")
    
    st.write("### - Analyse des corrélations entre les variables")
    st.write("En machine learning, une matrice de corrélation permet d'analyser les relations entre les variables explicatives (features) et aussi avec la variable cible. Cela aide à mieux comprendre les données et à optimiser la performance du modèle.")
    st.write("En effet, si deux variables explicatives sont fortement corrélées, elles apportent la même information. De la même manière, une variable qui a une corrélation très faible avec toutes les autres (y compris avec la variable cible) est peu utile pour la prédiction.", unsafe_allow_html=True)
    st.write("Suite à cette analyse, je supprime les variables <span style='color:red'>ATP et Best of</span>.", unsafe_allow_html=True)
    
    if st.checkbox("Afficher la Heatmap de corrélation"):
        st.write("### 🔥 Heatmap des corrélations")

    # Calcul de la matrice de corrélation
        numerical_cols = df.select_dtypes(include=['number'])  # Sélection des colonnes numériques
        correlation_matrix = numerical_cols.corr()

        # Création de la heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, linewidths=0.5, square=True, ax=ax)
        ax.set_title("Matrice de corrélation entre les variables", fontsize=14)

        st.pyplot(fig)

df = df.drop(columns=["ATP", "Best of"])

if page == pages[3] :

    st.write("### Informations sur le dataframe après nettoyage")
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_string = buffer.getvalue()

    # Affiche les informations dans un conteneur code
    st.code(info_string, language="text")

#df = df.drop(columns=["ATP", "Best of"])
X = df.drop(columns=["target"])  # Variables explicatives
y = df["target"]  # Variable cible
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
ohe = OneHotEncoder(drop='first', sparse_output= False, handle_unknown='ignore')
colonnes_a_encoder = ['Month', 'Surface', 'Court']
ohe.fit(X_train[colonnes_a_encoder])
X_train_encode = ohe.fit_transform(X_train[colonnes_a_encoder])
X_test_encode = ohe.transform(X_test[colonnes_a_encoder])
train_encoded_df = pd.DataFrame(X_train_encode, columns=ohe.get_feature_names_out(colonnes_a_encoder), index=X_train.index)
test_encoded_df = pd.DataFrame(X_test_encode, columns=ohe.get_feature_names_out(colonnes_a_encoder), index=X_test.index)
X_train = pd.concat([X_train.drop(columns=colonnes_a_encoder), train_encoded_df], axis=1)
X_test = pd.concat([X_test.drop(columns=colonnes_a_encoder), test_encoded_df], axis=1)
series_order = ['Grand Slam', 'Masters 1000', 'Masters', 'ATP 500', 'Masters Cup', 'International Gold', 'ATP 250', 'International']
round_order = ['The Final', 'Semifinals', 'Quarterfinals', '3rd Round', '2nd Round', '1st Round', 'Round Robin']
ordinal_encoder_series = OrdinalEncoder(categories=[series_order], handle_unknown='use_encoded_value', unknown_value=-1)
X_train['Series_encoded'] = ordinal_encoder_series.fit_transform(X_train[['Series']])
X_test['Series_encoded'] = ordinal_encoder_series.transform(X_test[['Series']])
ordinal_encoder_round = OrdinalEncoder(categories=[round_order], handle_unknown='use_encoded_value', unknown_value=-1)
X_train['Round_encoded'] = ordinal_encoder_round.fit_transform(X_train[['Round']])
X_test['Round_encoded'] = ordinal_encoder_round.transform(X_test[['Round']])
X_train = X_train.drop(columns=['Series', 'Round'])
X_test = X_test.drop(columns=['Series', 'Round'])
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred_dtc = dtc.predict(X_test)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.iloc[:, 0:4])
X_test_scaled = scaler.transform(X_test.iloc[:, 0:4])
X_train.iloc[:, 0:4] = X_train_scaled
X_test.iloc[:, 0:4] = X_test_scaled

reglog = LogisticRegression()
reglog.fit(X_train_scaled, y_train)
y_pred_reglog = rf.predict(X_test)

svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)

knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)


if page == pages[4] :
    st.title("Préparation pour l'entrainement des modèles")

    st.write("### Séparation des variables explicatives et de la variable cible en un jeu d'entrainement et un jeu de test")
    if st.checkbox("Afficher le code"):
        st.code("""
            X = df.drop(columns=["target"])  # Variables explicatives
            y = df["target"]  # Variable cible
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            """, language="python")
    
    st.write("### Encodage des variables catégorielles avec OneHotEncoder")
    if st.checkbox("Afficher le code de l'encodage"):
        st.code("""
            # Instanciation de l'encodeur
            ohe = OneHotEncoder(drop='first', sparse_output= False, handle_unknown='ignore')
            colonnes_a_encoder = ['Month', 'Surface', 'Court']
        
            # Encodage des variables catégorielles et transformation en DataFrame
            ohe.fit(X_train[colonnes_a_encoder])
            X_train_encode = ohe.fit_transform(X_train[colonnes_a_encoder])
            X_test_encode = ohe.transform(X_test[colonnes_a_encoder])
            train_encoded_df = pd.DataFrame(X_train_encode, columns=ohe.get_feature_names_out(colonnes_a_encoder), index=X_train.index)
            test_encoded_df = pd.DataFrame(X_test_encode, columns=ohe.get_feature_names_out(colonnes_a_encoder), index=X_test.index)
            X_train = pd.concat([X_train.drop(columns=colonnes_a_encoder), train_encoded_df], axis=1)
            X_test = pd.concat([X_test.drop(columns=colonnes_a_encoder), test_encoded_df], axis=1)
            """, language="python")

    st.write("### Encodage ordinal des variables avec une hiérarchie")
    if st.checkbox("Afficher le code de l'encodage ordinal"): 
        st.code("""
            # Définition des hiérarchies pour les variables 'Series' et 'Round'
            series_order = ['Grand Slam', 'Masters 1000', 'Masters', 'ATP 500', 'Masters Cup', 'International Gold', 'ATP 250', 'International']
            round_order = ['The Final', 'Semifinals', 'Quarterfinals', '3rd Round', '2nd Round', '1st Round', 'Round Robin']
        
            # Encodage ordinal de la variable Series
            ordinal_encoder_series = OrdinalEncoder(categories=[series_order], handle_unknown='use_encoded_value', unknown_value=-1)
            X_train['Series_encoded'] = ordinal_encoder_series.fit_transform(X_train[['Series']])
            X_test['Series_encoded'] = ordinal_encoder_series.transform(X_test[['Series']])
        
            # Encodage ordinal de la variable Round
            ordinal_encoder_round = OrdinalEncoder(categories=[round_order], handle_unknown='use_encoded_value', unknown_value=-1)
            X_train['Round_encoded'] = ordinal_encoder_round.fit_transform(X_train[['Round']])
            X_test['Round_encoded'] = ordinal_encoder_round.transform(X_test[['Round']])
            """, language="python")
    
    st.write("### Suppression des colonnes originales")
    if st.checkbox("Afficher le code de suppression"):
        st.code("""
            X_train = X_train.drop(columns=['Series', 'Round'])
            X_test = X_test.drop(columns=['Series', 'Round'])
            """, language="python")
    
    st.write("### Standardisation des données")
    if st.checkbox("Afficher le code de standardisation"):
        st.code("""
            # Instanciation du StandardScaler
            scaler = StandardScaler()
        
            # Standardisation des variables explicatives (les 4 premières colonnes)
            X_train_scaled = scaler.fit_transform(X_train.iloc[:, 0:4])
            X_test_scaled = scaler.transform(X_test.iloc[:, 0:4])
            X_train.iloc[:, 0:4] = X_train_scaled
            X_test.iloc[:, 0:4] = X_test_scaled
            """, language="python")
    
    st.write("### Sur-échantillonnage des données d'entraînement")
    if st.checkbox("Afficher le code du sur-échantillonnage"):
        st.code("""
            # Instanciation de l'objet SMOTE et application sur les données d'entraînement
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            """, language="python")

    st.write("### Taille de mes échantillons avant entraînement des modèles")
    st.write(f"**Taille du jeu d'entraînement** : {X_train.shape[0]} lignes dont {y_train.sum()} de la classe 1 et {y_train.shape[0] - y_train.sum()} de la classe 0")
    st.write(f"**Taille du jeu de test** : {X_test.shape[0]} lignes dont {y_test.sum()} de la classe 1 et {y_test.shape[0] - y_test.sum()} de la classe 0")
    
if page == pages[5] :
    st.write("### Machine learning")
    
    if st.checkbox("Afficher un aperçu du jeu d'entraînement") :
        st.dataframe(X_train.head(10))
    options=["Arbre de classification","KNN","SVM","Random Forest","Logistic Regression"]
    choix=st.selectbox("Selection de l'algorithme de machine learning", options)
    st.markdown(f"# **{choix}**")

    test_size = st.slider("Proportion du jeu de test", min_value=0.1, max_value=0.5, step=0.05, value=0.2)
    st.write(f"Le jeu de test représentera {test_size*100:.0f}% des données.")

# Division des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Modèle sélectionné
    if choix == "Random Forest":
        ohe = OneHotEncoder(drop='first', sparse_output= False, handle_unknown='ignore')
        colonnes_a_encoder = ['Month', 'Surface', 'Court']
        ohe.fit(X_train[colonnes_a_encoder])
        X_train_encode = ohe.fit_transform(X_train[colonnes_a_encoder])
        X_test_encode = ohe.transform(X_test[colonnes_a_encoder])
        train_encoded_df = pd.DataFrame(X_train_encode, columns=ohe.get_feature_names_out(colonnes_a_encoder), index=X_train.index)
        test_encoded_df = pd.DataFrame(X_test_encode, columns=ohe.get_feature_names_out(colonnes_a_encoder), index=X_test.index)
        X_train = pd.concat([X_train.drop(columns=colonnes_a_encoder), train_encoded_df], axis=1)
        X_test = pd.concat([X_test.drop(columns=colonnes_a_encoder), test_encoded_df], axis=1)
        series_order = ['Grand Slam', 'Masters 1000', 'Masters', 'ATP 500', 'Masters Cup', 'International Gold', 'ATP 250', 'International']
        round_order = ['The Final', 'Semifinals', 'Quarterfinals', '3rd Round', '2nd Round', '1st Round', 'Round Robin']
        ordinal_encoder_series = OrdinalEncoder(categories=[series_order], handle_unknown='use_encoded_value', unknown_value=-1)
        X_train['Series_encoded'] = ordinal_encoder_series.fit_transform(X_train[['Series']])
        X_test['Series_encoded'] = ordinal_encoder_series.transform(X_test[['Series']])
        ordinal_encoder_round = OrdinalEncoder(categories=[round_order], handle_unknown='use_encoded_value', unknown_value=-1)
        X_train['Round_encoded'] = ordinal_encoder_round.fit_transform(X_train[['Round']])
        X_test['Round_encoded'] = ordinal_encoder_round.transform(X_test[['Round']])
        X_train = X_train.drop(columns=['Series', 'Round'])
        X_test = X_test.drop(columns=['Series', 'Round'])
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
    #Entrainement du modèle
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Précision du modèle {choix} : {accuracy:.2f}")

    # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        st.write("### Matrice de confusion")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
    
    # Rapport de classification
        st.write("### Rapport de classification")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
        
# Modèle sélectionné
    if choix == "Arbre de classification":
        ohe = OneHotEncoder(drop='first', sparse_output= False, handle_unknown='ignore')
        colonnes_a_encoder = ['Month', 'Surface', 'Court']
        ohe.fit(X_train[colonnes_a_encoder])
        X_train_encode = ohe.fit_transform(X_train[colonnes_a_encoder])
        X_test_encode = ohe.transform(X_test[colonnes_a_encoder])
        train_encoded_df = pd.DataFrame(X_train_encode, columns=ohe.get_feature_names_out(colonnes_a_encoder), index=X_train.index)
        test_encoded_df = pd.DataFrame(X_test_encode, columns=ohe.get_feature_names_out(colonnes_a_encoder), index=X_test.index)
        X_train = pd.concat([X_train.drop(columns=colonnes_a_encoder), train_encoded_df], axis=1)
        X_test = pd.concat([X_test.drop(columns=colonnes_a_encoder), test_encoded_df], axis=1)
        series_order = ['Grand Slam', 'Masters 1000', 'Masters', 'ATP 500', 'Masters Cup', 'International Gold', 'ATP 250', 'International']
        round_order = ['The Final', 'Semifinals', 'Quarterfinals', '3rd Round', '2nd Round', '1st Round', 'Round Robin']
        ordinal_encoder_series = OrdinalEncoder(categories=[series_order], handle_unknown='use_encoded_value', unknown_value=-1)
        X_train['Series_encoded'] = ordinal_encoder_series.fit_transform(X_train[['Series']])
        X_test['Series_encoded'] = ordinal_encoder_series.transform(X_test[['Series']])
        ordinal_encoder_round = OrdinalEncoder(categories=[round_order], handle_unknown='use_encoded_value', unknown_value=-1)
        X_train['Round_encoded'] = ordinal_encoder_round.fit_transform(X_train[['Round']])
        X_test['Round_encoded'] = ordinal_encoder_round.transform(X_test[['Round']])
        X_train = X_train.drop(columns=['Series', 'Round'])
        X_test = X_test.drop(columns=['Series', 'Round'])
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
    #Entrainement du modèle
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Précision du modèle {choix} : {accuracy:.2f}")

    # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        st.write("### Matrice de confusion")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
    
    # Rapport de classification
        st.write("### Rapport de classification")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
        
    if choix == "KNN":
        ohe = OneHotEncoder(drop='first', sparse_output= False, handle_unknown='ignore')
        colonnes_a_encoder = ['Month', 'Surface', 'Court']
        ohe.fit(X_train[colonnes_a_encoder])
        X_train_encode = ohe.fit_transform(X_train[colonnes_a_encoder])
        X_test_encode = ohe.transform(X_test[colonnes_a_encoder])
        train_encoded_df = pd.DataFrame(X_train_encode, columns=ohe.get_feature_names_out(colonnes_a_encoder), index=X_train.index)
        test_encoded_df = pd.DataFrame(X_test_encode, columns=ohe.get_feature_names_out(colonnes_a_encoder), index=X_test.index)
        X_train = pd.concat([X_train.drop(columns=colonnes_a_encoder), train_encoded_df], axis=1)
        X_test = pd.concat([X_test.drop(columns=colonnes_a_encoder), test_encoded_df], axis=1)
        series_order = ['Grand Slam', 'Masters 1000', 'Masters', 'ATP 500', 'Masters Cup', 'International Gold', 'ATP 250', 'International']
        round_order = ['The Final', 'Semifinals', 'Quarterfinals', '3rd Round', '2nd Round', '1st Round', 'Round Robin']
        ordinal_encoder_series = OrdinalEncoder(categories=[series_order], handle_unknown='use_encoded_value', unknown_value=-1)
        X_train['Series_encoded'] = ordinal_encoder_series.fit_transform(X_train[['Series']])
        X_test['Series_encoded'] = ordinal_encoder_series.transform(X_test[['Series']])
        ordinal_encoder_round = OrdinalEncoder(categories=[round_order], handle_unknown='use_encoded_value', unknown_value=-1)
        X_train['Round_encoded'] = ordinal_encoder_round.fit_transform(X_train[['Round']])
        X_test['Round_encoded'] = ordinal_encoder_round.transform(X_test[['Round']])
        X_train = X_train.drop(columns=['Series', 'Round'])
        X_test = X_test.drop(columns=['Series', 'Round'])
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.iloc[:, 0:4])
        X_test_scaled = scaler.transform(X_test.iloc[:, 0:4])
        X_train.iloc[:, 0:4] = X_train_scaled
        X_test.iloc[:, 0:4] = X_test_scaled
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
    #Entrainement du modèle
        k = st.slider("Choisissez le nombre de voisins (k)", min_value=1, max_value=5, step=1, value=5)
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Précision du modèle {choix} : {accuracy:.2f}")

    # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        st.write("### Matrice de confusion")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
    
    # Rapport de classification
        st.write("### Rapport de classification")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
        
    if choix == "SVM":
        ohe = OneHotEncoder(drop='first', sparse_output= False, handle_unknown='ignore')
        colonnes_a_encoder = ['Month', 'Surface', 'Court']
        ohe.fit(X_train[colonnes_a_encoder])
        X_train_encode = ohe.fit_transform(X_train[colonnes_a_encoder])
        X_test_encode = ohe.transform(X_test[colonnes_a_encoder])
        train_encoded_df = pd.DataFrame(X_train_encode, columns=ohe.get_feature_names_out(colonnes_a_encoder), index=X_train.index)
        test_encoded_df = pd.DataFrame(X_test_encode, columns=ohe.get_feature_names_out(colonnes_a_encoder), index=X_test.index)
        X_train = pd.concat([X_train.drop(columns=colonnes_a_encoder), train_encoded_df], axis=1)
        X_test = pd.concat([X_test.drop(columns=colonnes_a_encoder), test_encoded_df], axis=1)
        series_order = ['Grand Slam', 'Masters 1000', 'Masters', 'ATP 500', 'Masters Cup', 'International Gold', 'ATP 250', 'International']
        round_order = ['The Final', 'Semifinals', 'Quarterfinals', '3rd Round', '2nd Round', '1st Round', 'Round Robin']
        ordinal_encoder_series = OrdinalEncoder(categories=[series_order], handle_unknown='use_encoded_value', unknown_value=-1)
        X_train['Series_encoded'] = ordinal_encoder_series.fit_transform(X_train[['Series']])
        X_test['Series_encoded'] = ordinal_encoder_series.transform(X_test[['Series']])
        ordinal_encoder_round = OrdinalEncoder(categories=[round_order], handle_unknown='use_encoded_value', unknown_value=-1)
        X_train['Round_encoded'] = ordinal_encoder_round.fit_transform(X_train[['Round']])
        X_test['Round_encoded'] = ordinal_encoder_round.transform(X_test[['Round']])
        X_train = X_train.drop(columns=['Series', 'Round'])
        X_test = X_test.drop(columns=['Series', 'Round'])
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.iloc[:, 0:4])
        X_test_scaled = scaler.transform(X_test.iloc[:, 0:4])
        X_train.iloc[:, 0:4] = X_train_scaled
        X_test.iloc[:, 0:4] = X_test_scaled
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
    #Entrainement du modèle
        model = SVC()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Précision du modèle {choix} : {accuracy:.2f}")

    # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        st.write("### Matrice de confusion")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
    
    # Rapport de classification
        st.write("### Rapport de classification")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
     
    if choix == "Logistic Regression":
        ohe = OneHotEncoder(drop='first', sparse_output= False, handle_unknown='ignore')
        colonnes_a_encoder = ['Month', 'Surface', 'Court']
        ohe.fit(X_train[colonnes_a_encoder])
        X_train_encode = ohe.fit_transform(X_train[colonnes_a_encoder])
        X_test_encode = ohe.transform(X_test[colonnes_a_encoder])
        train_encoded_df = pd.DataFrame(X_train_encode, columns=ohe.get_feature_names_out(colonnes_a_encoder), index=X_train.index)
        test_encoded_df = pd.DataFrame(X_test_encode, columns=ohe.get_feature_names_out(colonnes_a_encoder), index=X_test.index)
        X_train = pd.concat([X_train.drop(columns=colonnes_a_encoder), train_encoded_df], axis=1)
        X_test = pd.concat([X_test.drop(columns=colonnes_a_encoder), test_encoded_df], axis=1)
        series_order = ['Grand Slam', 'Masters 1000', 'Masters', 'ATP 500', 'Masters Cup', 'International Gold', 'ATP 250', 'International']
        round_order = ['The Final', 'Semifinals', 'Quarterfinals', '3rd Round', '2nd Round', '1st Round', 'Round Robin']
        ordinal_encoder_series = OrdinalEncoder(categories=[series_order], handle_unknown='use_encoded_value', unknown_value=-1)
        X_train['Series_encoded'] = ordinal_encoder_series.fit_transform(X_train[['Series']])
        X_test['Series_encoded'] = ordinal_encoder_series.transform(X_test[['Series']])
        ordinal_encoder_round = OrdinalEncoder(categories=[round_order], handle_unknown='use_encoded_value', unknown_value=-1)
        X_train['Round_encoded'] = ordinal_encoder_round.fit_transform(X_train[['Round']])
        X_test['Round_encoded'] = ordinal_encoder_round.transform(X_test[['Round']])
        X_train = X_train.drop(columns=['Series', 'Round'])
        X_test = X_test.drop(columns=['Series', 'Round'])
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.iloc[:, 0:4])
        X_test_scaled = scaler.transform(X_test.iloc[:, 0:4])
        X_train.iloc[:, 0:4] = X_train_scaled
        X_test.iloc[:, 0:4] = X_test_scaled
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
    #Entrainement du modèle
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Précision du modèle {choix} : {accuracy:.2f}")

    # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        st.write("### Matrice de confusion")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
    
    # Rapport de classification
        st.write("### Rapport de classification")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

model_metrics = {}

# Liste des modèles et de leurs prédictions
models = {
    "Arbre de classification": y_pred_dtc,
    "Random Forest": y_pred_rf,
    "Régression logistique": y_pred_reglog,
    "SVM": y_pred_svm,
    "KNN": y_pred_knn,  
    }

# Calcul des métriques pour chaque modèle
for model_name, y_pred in models.items():
    model_metrics[model_name] = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 3),
        "Precision": round(precision_score(y_test, y_pred), 3),
        "Recall": round(recall_score(y_test, y_pred), 3),
        "F1-Score": round(f1_score(y_test, y_pred), 3)
        }

# Création du DataFrame
df_metrics = pd.DataFrame.from_dict(model_metrics, orient='index')
    
if page == pages[5] :
    st.write("### Tableau récapitulatif des métriques par modèle (performances sur la classe positive : victoire surprise)")
    st.dataframe(df_metrics)

proba_rf = rf.predict_proba(X_test)[:, 1]
proba_reglog = reglog.predict_proba(X_test_scaled)[:, 1]
proba_knn = knn_model.predict_proba(X_test_scaled)[:, 1]
proba_dtc = dtc.predict_proba(X_test)[:, 1]
proba_svm = svm_model.predict_proba(X_test_scaled)[:, 1]  # Seulement si SVC(probability=True)

df_results = pd.DataFrame({
    'Actual_Winner': y_test,  # Vraie issue du match
    'proba_RandomForest': proba_rf,
    'proba_LogReg': proba_reglog.round(2),
    'proba_KNN': proba_knn,
    'proba_Tree': proba_dtc,
    'proba_SVM': proba_svm.round(2),
    })

# Comparaison des probabilités de victoire des bookmakers et du modèle Random Forest
# récupération des données d'origine
df1 = pd.read_csv('atp_data.csv')
df1['target'] = df.apply(lambda row: target(row), axis = 1)

# Extraire les cotes des bookmakers pour les matchs de test et calcul de la probabilité associée
cotes_bookmakers = df1.loc[X_test.index, ['B365W', 'B365L', 'PSL', 'PSW', 'target', 'proba_elo']]
cotes_bookmakers['proba_PSL']= 1/cotes_bookmakers['PSL']
cotes_bookmakers['proba_B365W']= 1/cotes_bookmakers['B365W']
cotes_bookmakers['proba_PSW']= 1/cotes_bookmakers['PSW']
cotes_bookmakers['proba_B365L']= 1/cotes_bookmakers['B365L']

# Concaténation des probabilités des bookmakers et du modèle Random Forest
proba_rf_df = pd.DataFrame(df_results, columns=['proba_RandomForest'])
df_comparison = pd.concat([cotes_bookmakers, proba_rf_df], axis=1)

#Extraire les données lorsque target == 1
df_target_1 = df_comparison[df_comparison['target'] == 1]
df_target_1 = df_target_1.drop(columns=['target', 'B365L', 'PSL', 'proba_PSL', 'proba_B365L'], axis=1)

if page == pages[6] :
    st.write("### Calcul des probabilités")
    
    st.write("Calcul des probabilités que le joueur le moins bien classé gagne pour chaque modèle (victoire surprise) :")
    st.dataframe(df_results.head(10))

    st.write("Comparaison des probabilités de victoire des bookmakers et celle du Random Forest (victoire surprise) :")
    st.dataframe(df_target_1.head(10))


if page == pages[7] :
    
    pct_RF_vs_B365W = (df_target_1['proba_RandomForest'] > df_target_1['proba_B365W']).mean() * 100
    pct_RF_vs_PSW = (df_target_1['proba_RandomForest'] > df_target_1['proba_PSW']).mean() * 100

    st.write("### Conclusion")
    st.write("Suite à la comparaison des cas pour lesquels la probabilité du Random Forest d’obtenir une victoire surprise (lorsque c’est réellement le cas, c’est à dire lorsque target = 1) est supérieure à celles données par les côtes des bookmakers, j'obtiens les résultats suivants :") 
    st.write(f"- Le modèle Random Forest est meilleur que le bookmaker Bet365 dans **{pct_RF_vs_B365W:.2f}%** des cas")
    st.write(f"- Le modèle Random Forest est meilleur que le bookmaker Pinnacles Sport dans **{pct_RF_vs_PSW:.2f}%** des cas")
    st.write("**Je n’obtiens pas de meilleurs résultats que les bookmakers** dans plus de la moitié des cas comme je le souhaitais au départ.")
    st.write("La stratégie pourrait être encore affinée en établissant un seuil décisionnel pour maximiser le rapport risque/rendement, potentiellement en se concentrant sur les matchs où mon modèle attribue une probabilité supérieure à 0.45 à une surprise, même lorsque cette valeur reste inférieure à 0.5.")
    st.write("Pour battre les bookmakers, je devrais envisager d'intégrer des facteurs complémentaires (variance historique des performances, les spécificités des surfaces, l'historique des confrontations directes, ou l'état de forme récent) afin de construire un système de paris sophistiqué capable d'exploiter systématiquement les inefficiences du marché.")
