# 🌍 Child Labor Analysis: Clustering & Trends

## 📋 Description
Ce projet vise à analyser les dynamiques mondiales du travail des enfants en utilisant des techniques de **Data Mining** non supervisées. 
À partir des données de l'OIT (Organisation Internationale du Travail), nous identifions des groupes de pays aux comportements similaires pour orienter les politiques publiques.

**Contexte :** Projet Master 1

## 🛠 Méthodologie
1. **ETL & Preprocessing :** - Nettoyage des données brutes.
   - Pivot et imputation pour créer des séries temporelles complètes par pays.
2. **Feature Engineering :**
   - Calcul des taux moyens et des pentes d'évolution (Trend).
3. **Clustering (K-Means) :**
   - Normalisation (StandardScaler).
   - Optimisation du nombre de clusters via la **Méthode du Coude (Elbow)** et le **Score Silhouette**.

## 📊 Résultats Clés
L'algorithme a permis d'isoler 3 profils types de pays :
* **Cluster A (Critique) :** Taux élevés et stagnation.
* **Cluster B (En transition) :** Taux moyens mais en forte baisse.
* **Cluster C (Sous contrôle) :** Taux faibles et stables.

![Résultats du Clustering](clusters_result.png)

## 💻 Installation et Exécution

1. Cloner le projet :
```bash
git clone [https://github.com/ton-user/child-labor-clustering.git](https://github.com/ton-user/child-labor-clustering.git)
