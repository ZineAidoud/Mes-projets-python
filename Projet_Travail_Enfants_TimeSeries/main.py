"""
=============================================================================
TITRE : Analyse et Clustering du Travail des Enfants (K-Means & Time Series)
AUTEUR : Zine Elabidine AIDOUD & Rayane Hocine ARHAB
PROJET : Master 1 - Data Mining & Séries Temporelles
DATE : Mai 2025

DESCRIPTION :
Ce script analyse les données de l'OIT (ILO) pour segmenter les pays selon 
l'évolution du travail des enfants.
1. ETL : Nettoyage et pivot des données temporelles.
2. Clustering : K-Means (avec méthode Elbow & Silhouette pour k optimal).
3. Analyse : Visualisation des clusters (Pays à risque vs En amélioration).
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings

# Configuration
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

def charger_et_nettoyer(filepath):
    """
    Charge le dataset et prépare la matrice Pays x Années.
    """
    print("--- 1. Chargement des données ---")
    try:
        # Lecture du CSV (séparateur point-virgule selon ton fichier)
        df = pd.read_csv(filepath, sep=';')
        
        # Sélection des colonnes utiles comme dans ton notebook
        cols_utiles = ['ref_area.label', 'time', 'obs_value']
        df = df[cols_utiles]
        
        # Nettoyage des valeurs non numériques
        df['obs_value'] = pd.to_numeric(df['obs_value'], errors='coerce')
        df = df.dropna()
        
        # Pivot pour avoir les séries temporelles (Lignes=Pays, Colonnes=Années)
        df_pivot = df.pivot_table(index='ref_area.label', columns='time', values='obs_value')
        
        # Interpolation pour gérer les années manquantes (très important pour le clustering)
        df_clean = df_pivot.interpolate(method='linear', axis=1, limit_direction='both')
        df_clean = df_clean.dropna() # On supprime ceux qui restent vides
        
        print(f"Données prêtes : {df_clean.shape[0]} pays analysés sur {df_clean.shape[1]} années.")
        return df_clean

    except Exception as e:
        print(f"Erreur de chargement : {e}")
        return None

def determiner_k_optimal(X_scaled):
    """
    Affiche la méthode du coude (Elbow) et le score Silhouette 
    pour choisir le bon nombre de clusters.
    """
    print("\n--- 2. Recherche du nombre optimal de clusters (k) ---")
    inertia = []
    silhouette_scores = []
    K_range = range(2, 10)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

    # Graphique couplé (Elbow + Silhouette)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Nombre de clusters (k)')
    ax1.set_ylabel('Inertie (Elbow)', color=color)
    ax1.plot(K_range, inertia, 'o-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Score Silhouette', color=color)
    ax2.plot(K_range, silhouette_scores, 's--', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Méthode du Coude et Score Silhouette")
    plt.tight_layout()
    plt.savefig("optimization_k.png")
    print("Graphique d'optimisation généré : 'optimization_k.png'")

def appliquer_clustering(df_clean, n_clusters=3):
    """
    Applique K-Means et génère les visualisations.
    """
    print(f"\n--- 3. Clustering K-Means (k={n_clusters}) ---")
    
    # Feature Engineering : On utilise la Moyenne et la Tendance
    features = pd.DataFrame(index=df_clean.index)
    features['Moyenne'] = df_clean.mean(axis=1)
    # Tendance : Dernière année - Première année
    features['Evolution'] = df_clean.iloc[:, -1] - df_clean.iloc[:, 0]
    
    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    features['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Visualisation des clusters
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=features, x='Moyenne', y='Evolution', 
        hue='Cluster', palette='viridis', s=100, style='Cluster'
    )
    
    # Annoter quelques pays pour l'exemple
    for i in range(features.shape[0]):
        if i % 5 == 0: # Annoter un pays sur 5 pour ne pas surcharger
            plt.text(
                features.Moyenne[i]+0.2, 
                features.Evolution[i], 
                features.index[i], 
                fontsize=9, alpha=0.7
            )

    plt.title(f"Segmentation des Pays : Taux Moyen vs Évolution ({df_clean.columns[0]}-{df_clean.columns[-1]})")
    plt.xlabel("Taux Moyen de Travail des Enfants (%)")
    plt.ylabel("Évolution (Points de pourcentage)")
    plt.axhline(0, color='grey', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.savefig("clusters_result.png")
    print("Graphique des clusters généré : 'clusters_result.png'")
    
    # Affichage console des groupes
    print("\n--- Analyse des Groupes ---")
    groupe_moyen = features.groupby('Cluster')[['Moyenne', 'Evolution']].mean()
    print(groupe_moyen)
    
    return features

if __name__ == "__main__":
    fichier_csv = "dataset.csv" # Assure-toi de renommer ton fichier
    
    # Pipeline complet
    df = charger_et_nettoyer(fichier_csv)
    
    if df is not None:
        # Préparation pour le choix de K
        # On utilise juste la moyenne et l'évolution pour simplifier le graph 2D
        features_temp = pd.DataFrame()
        features_temp['Moyenne'] = df.mean(axis=1)
        features_temp['Evolution'] = df.iloc[:, -1] - df.iloc[:, 0]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features_temp)
        
        # 1. Trouver le bon K
        determiner_k_optimal(X_scaled)
        
        # 2. Lancer le clustering (k=3 semble souvent bon pour ce type de données)
        appliquer_clustering(df, n_clusters=3)
        
        print("\n=== Analyse terminée ===")
