import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix

# Titre de l'application Streamlit
st.title("Logique_Floue: Fuzzification")

# Initialisation des variables de session pour les données nettoyées et fuzzifiées
if "dataset_cleaned" not in st.session_state:
    st.session_state.dataset_cleaned = None
if "dataset_fuzifie" not in st.session_state:
    st.session_state.dataset_fuzifie = None
if "scaler" not in st.session_state:
    st.session_state.scaler = StandardScaler()

# Fonction pour charger différents types de fichiers (CSV, Excel, JSON)
def load_data(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == "csv":
        return pd.read_csv(uploaded_file)
    elif file_extension == "xlsx":
        return pd.read_excel(uploaded_file)
    elif file_extension == "json":
        return pd.read_json(uploaded_file)
    else:
        st.error("Format de fichier non supporté. Veuillez télécharger un fichier CSV, Excel ou JSON.")
        return None

# Chargement des données
uploaded_file = st.file_uploader("Charger le dataset", type=["csv", "xlsx", "json"])
if uploaded_file is not None:
    dataset_original = load_data(uploaded_file)
    
    if dataset_original is not None:
        st.write("### Données Originales")
        st.write(dataset_original.describe())  # Statistiques descriptives
        st.write(dataset_original)  # Affichage du dataset

        # Affichage des histogrammes combinés avec les courbes de distribution
        st.write("### Histogrammes des données originales avec courbes de distribution")
        for col in dataset_original.select_dtypes(include=[np.number]).columns:
            plt.figure(figsize=(10, 6))
            
            # Histogramme et courbe de distribution combinés
            sns.histplot(dataset_original[col], bins=20, kde=True)
            plt.title(f"Histogramme et distribution de {col}")
            plt.xlabel(col)
            st.pyplot(plt)

    # Nettoyage des données
    if st.button("Nettoyer les Données"):
        dataset_cleaned = dataset_original.copy()

        # Encoder les colonnes catégoriques avec LabelEncoder
        for col in dataset_cleaned.select_dtypes(include=['object']).columns:
            encoder = LabelEncoder()
            dataset_cleaned[col] = encoder.fit_transform(dataset_cleaned[col].astype(str))

        # Remplacement des valeurs manquantes par la moyenne
        imputer = SimpleImputer(strategy="mean")
        dataset_cleaned.iloc[:, :] = imputer.fit_transform(dataset_cleaned)

        # Appliquer la mise à l'échelle des données (StandardScaler)
        st.session_state.scaler.fit(dataset_cleaned.iloc[:, :-1])
        dataset_cleaned.iloc[:, :-1] = st.session_state.scaler.transform(dataset_cleaned.iloc[:, :-1])

        st.session_state.dataset_cleaned = dataset_cleaned
        st.write("### Données Nettoyées")
        st.write(dataset_cleaned.describe())  # Statistiques descriptives sur les données nettoyées
        st.write(dataset_cleaned)  # Affichage du dataset nettoyé

        # Affichage des histogrammes combinés avec les courbes de distribution pour les données nettoyées
        st.write("### Histogrammes des données nettoyées avec courbes de distribution")
        for col in dataset_cleaned.select_dtypes(include=[np.number]).columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(dataset_cleaned[col], bins=20, kde=True)
            plt.title(f"Histogramme et distribution de {col} après nettoyage")
            plt.xlabel(col)
            st.pyplot(plt)

    # Fuzzification des données
    if st.button("Fuzifier les Données"):
        if st.session_state.dataset_cleaned is not None:
            # Ajouter du bruit aléatoire à toutes les colonnes (fuzzification)
            noise = np.random.normal(0, 0.1, st.session_state.dataset_cleaned.shape)
            dataset_fuzifie = st.session_state.dataset_cleaned + noise
            st.session_state.dataset_fuzifie = dataset_fuzifie

            st.write("### Données Fuzifiées")
            st.write(dataset_fuzifie.describe())  # Statistiques descriptives des données fuzzifiées
            st.write(dataset_fuzifie)  # Affichage du dataset fuzzifié

            # Affichage des histogrammes combinés avec les courbes de distribution pour les données fuzzifiées
            st.write("### Histogrammes des données fuzzifiées avec courbes de distribution")
            for col in dataset_fuzifie.select_dtypes(include=[np.number]).columns:
                plt.figure(figsize=(10, 6))
                sns.histplot(dataset_fuzifie[col], bins=20, kde=True)
                plt.title(f"Histogramme et distribution de {col} après fuzzification")
                plt.xlabel(col)
                st.pyplot(plt)
        else:
            st.error("Veuillez d'abord nettoyer les données avant de les fuzzifier.")

    # Sélection du modèle
    model_type = st.selectbox("Choisir un modèle", ["Gradient Boosting", "Random Forest", "SVM Classification", "SVR Régression"])

    def normalize_error(errors, y_true, error_type="mae"):
        # Calcul de la plage des valeurs réelles (max - min)
        y_range = y_true.max() - y_true.min()

        if error_type == "mae":
            return errors / y_range
        elif error_type == "mse":
            return errors / (y_range ** 2)  # Diviser par la plage au carré pour normaliser la MSE

        return errors


    # Fonction d'entraînement du modèle
    def train_model(data, model_type, label):
        if data is None:
            st.error(f"Aucune donnée disponible pour l'entraînement ({label}).")
            return

        # Séparation des caractéristiques et de la cible
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Choix du modèle en fonction de l'option de l'utilisateur
        if model_type == "Gradient Boosting":
            model = GradientBoostingRegressor()
        elif model_type == "Random Forest":
            model = RandomForestRegressor()
        elif model_type == "SVM Classification":
            model = SVC()
        elif model_type == "SVR Régression":
            model = SVR()

        # Entraînement du modèle
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write(f"### Résultats du modèle {model_type} sur {label}")

        if model_type in ["Gradient Boosting", "Random Forest", "SVR Régression"]:
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Normalisation des erreurs
            mae_normalized = normalize_error(mae, y_test, error_type="mae")
            mse_normalized = normalize_error(mse, y_test, error_type="mse")

            st.write(f"MAE normalisé: {mae_normalized:.2f}, MSE normalisé: {mse_normalized:.2f}, R²: {r2:.2f}")

            # Graphique y_test vs y_pred
            plt.figure(figsize=(6, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r')
            plt.xlabel("Valeurs Réelles")
            plt.ylabel("Prédictions")
            plt.title(f"Comparaison y_test vs y_pred - {model_type}")
            st.pyplot(plt)
        else:
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            st.write(f"Précision: {accuracy:.2f}")

            # Heatmap de la matrice de confusion
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Prédictions")
            plt.ylabel("Réel")
            plt.title("Matrice de confusion")
            st.pyplot(plt)

    # Entraîner avec les données nettoyées
    if st.button("Entraîner Modèle Nettoyé"):
        if st.session_state.dataset_cleaned is not None:
            train_model(st.session_state.dataset_cleaned, model_type, "données nettoyées")
        else:
            st.error("Veuillez nettoyer les données avant de les entraîner.")

    # Entraîner avec les données fuzzifiées
    if st.button("Entraîner Modèle Fuzifié"):
        if st.session_state.dataset_fuzifie is not None:
            train_model(st.session_state.dataset_fuzifie, model_type, "données fuzzifiées")
        else:
            st.error("Veuillez fuzzifier les données avant de les entraîner.")
